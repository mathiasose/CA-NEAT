import sqlite3
from operator import attrgetter
from random import choice
from statistics import median, mean
from typing import Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

import kombu.exceptions
import psycopg2
import sqlalchemy.exc
from celery.canvas import chain
from neat.genome import Genome
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype
from neat.species import Species

from ca_neat.ca.analysis import calculate_lambda
from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.database import Individual, Scenario, get_db
from ca_neat.ga.population import create_initial_population, neat_reproduction, sort_into_species, speciate
from ca_neat.ga.selection import PAIR_SELECTION_F_T
from ca_neat.ga.serialize import deserialize_gt, serialize_gt
from ca_neat.report import send_message_via_pushbullet
from ca_neat.utils import pluck
from celery_app import app

FITNESS_F_T = Callable[[FeedForwardNetwork, CAConfig], float]

AUTO_RETRY = {
    'autoretry_for': (sqlalchemy.exc.OperationalError, sqlite3.OperationalError, psycopg2.OperationalError),
    'retry_kwargs': {'countdown': 5},
}


class SpeciesSet:
    """
    Encapsulates data being transferred between tasks
    """

    def __init__(self, species, scenario):
        self.species = set(species)
        self.scenario = scenario

    def __iter__(self):
        return iter(self.species)

    def __len__(self):
        return len(self.species)

    def __repr__(self):
        sizes = [len(s.members) for s in self.species]
        ages = [s.age for s in self.species]
        return f'{self.scenario}, ' \
               f'{len(self)} species, ' \
               f'mean age {mean(ages):.1f}, ' \
               f'mean size {mean(sizes):.1f}, ' \
               f'median size {median(sizes):.1f}'


def initialize_scenario(db_path: str, description: str, fitness_f: FITNESS_F_T, pair_selection_f: PAIR_SELECTION_F_T,
                        neat_config: CPPNNEATConfig, ca_config: CAConfig,
                        initial_genotypes: Optional[List[Genome]] = None) -> Scenario:
    print(db_path)

    db = get_db(db_path)
    session = db.Session()
    scenario = db.save_scenario(scenario=Scenario(
        description=description,
        generations=neat_config.generations,
        population_size=neat_config.pop_size
    ), session=session)
    session.commit()

    assert scenario

    if not initial_genotypes:
        initial_genotypes = list(create_initial_population(neat_config))

    if neat_config.do_speciation:
        speciate(initial_genotypes, compatibility_threshold=neat_config.compatibility_threshold)
    else:
        for gt in initial_genotypes:
            gt.species_id = 1

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario.id,
        generation=0,
        genotypes=initial_genotypes,
        fitness_f=fitness_f,
        pair_selection_f=pair_selection_f,
        neat_config=neat_config,
        ca_config=ca_config,
    )
    session.close()

    return scenario


def initialize_generation(db_path: str, scenario_id: int, generation: int, genotypes: List[Genome],
                          pair_selection_f: PAIR_SELECTION_F_T, fitness_f: FITNESS_F_T, neat_config: CPPNNEATConfig,
                          ca_config: CAConfig) -> None:
    from celery import group, chord

    grouped_tasks = group(handle_individual.s(
        scenario_id=scenario_id,
        generation=generation,
        individual_number=i,
        genotype=genotype,
        fitness_f=fitness_f,
        ca_config=ca_config,
    ) for i, genotype in enumerate(genotypes))

    final_task = persist_results.subtask(
        args=(db_path, scenario_id, generation, fitness_f, pair_selection_f, neat_config, ca_config),
    )

    chord(grouped_tasks, final_task)()


@app.task(name='persist_results', bind=True, **AUTO_RETRY)
def persist_results(task, results, db_path: str, scenario_id: int, generation_n: int, fitness_f: FITNESS_F_T,
                    pair_selection_f: PAIR_SELECTION_F_T, neat_config: CPPNNEATConfig, ca_config: CAConfig) -> str:
    db = get_db(db_path)
    session = db.Session()
    session.bulk_save_objects(results)
    session.commit()
    session.close()

    check_if_done.delay(
        db_path=db_path,
        scenario_id=scenario_id,
        generation_n=generation_n,
        fitness_f=fitness_f,
        pair_selection_f=pair_selection_f,
        neat_config=neat_config,
        ca_config=ca_config
    )

    return 'Scenario {scenario_id}, generation {generation_n}, {n_individuals} individuals, max fitness {max_f:.2f}'.format(
        scenario_id=scenario_id,
        generation_n=generation_n,
        n_individuals=len(results),
        max_f=max(pluck(results, 'fitness'))
    )


@app.task(name='check_if_done', bind=True, **AUTO_RETRY)
def check_if_done(task, db_path: str, scenario_id: int, generation_n: int, fitness_f: FITNESS_F_T,
                  pair_selection_f: PAIR_SELECTION_F_T, neat_config: CPPNNEATConfig, ca_config: CAConfig) -> str:
    db = get_db(db_path)
    session = db.Session()

    scenario = db.get_scenario(scenario_id, session=session)
    population = db.get_generation(scenario_id, generation_n, session=session)

    next_gen = generation_n + 1

    optimal_solution = population.filter(Individual.fitness >= 1.0)
    optimal_found = (neat_config.stop_when_optimal_found and session.query(optimal_solution.exists()).scalar())

    session.close()

    if (next_gen >= scenario.generations) or optimal_found:
        return '{} finished after {} generations'.format(
            scenario,
            next_gen,
        )
    else:
        chain(
            reproduction_io.subtask(
                args=(db_path, scenario_id, generation_n, neat_config),
            ),
            reproduction.subtask(
                args=(db_path, scenario_id, generation_n, fitness_f, pair_selection_f, neat_config, ca_config),
            )
        ).apply_async()

        return '{scenario}, generation {generation_n}'.format(
            scenario=scenario,
            generation_n=generation_n,
        )


@app.task(name='reproduction_io', bind=True, **AUTO_RETRY)
def reproduction_io(task, db_path: str, scenario_id: int, generation_n: int, neat_config: CPPNNEATConfig) \
        -> Tuple[Scenario, Set[Species]]:
    db = get_db(db_path)
    session = db.Session()

    scenario = db.get_scenario(scenario_id, session=session)
    population = db.get_generation(scenario_id, generation_n, session=session) \
        .with_entities(Individual.genotype)

    species = sort_into_species(
        genotypes=(
            deserialize_gt(
                gt_json_bytes=individual.genotype,
                neat_config=neat_config
            ) for individual in population
        )
    )

    stagnation_limit = neat_config.stagnation_limit
    stagnation_cutoff = generation_n - stagnation_limit

    if (not neat_config.do_speciation) or (not stagnation_limit) or (generation_n < stagnation_limit):
        alive_species = SpeciesSet(species, scenario)
    else:
        all_individuals = db \
            .get_individuals(scenario_id=scenario_id, session=session) \
            .filter(generation_n >= stagnation_cutoff) \
            .with_entities(Individual.fitness)

        stagnant: Dict[int, bool] = {}
        medians: Dict[int, float] = {}
        for s in species:
            individuals = all_individuals \
                .filter(Individual.species == UUID(int=s.ID)) \
                .order_by(Individual.generation, Individual.fitness)

            medians[s.ID] = median(pluck(individuals, 'fitness'))

            s.age = generation_n - db.get_species_birth_generation(scenario_id, s.ID, session=session)

            if s.age < stagnation_limit:
                stagnant[s.ID] = False
            else:
                first = individuals.first()
                last = individuals.order_by(-Individual.generation, Individual.fitness).first()
                stagnant[s.ID] = first.fitness == last.fitness

        median_of_medians = median(medians.values())

        for s in species:
            if medians[s.ID] >= median_of_medians:
                stagnant[s.ID] = False

        if len(species) > 1 \
                and not any(stagnant.values()) \
                and all(s.age > stagnation_limit for s in species):
            oldest_ID = sorted(species, key=attrgetter('age'))[-1].ID
            stagnant[oldest_ID] = True

        alive_species = SpeciesSet(
            species=(s for s in species if not stagnant[s.ID]),
            scenario=scenario
        )

        if len(alive_species) < len(species):
            print(f'Scenario {scenario_id}: removed {len(species) - len(alive_species)} species')

    session.close()

    return alive_species


@app.task(name='reproduction', bind=True, **AUTO_RETRY)
def reproduction(task, results, db_path: str, scenario_id: int, generation_n: int, fitness_f: FITNESS_F_T,
                 pair_selection_f: PAIR_SELECTION_F_T, neat_config: CPPNNEATConfig, ca_config: CAConfig) -> str:
    alive_species = results.species
    scenario = results.scenario

    next_gen_species, next_gen_genotypes = neat_reproduction(
        species=alive_species,
        pop_size=scenario.population_size,
        survival_threshold=neat_config.survival_threshold,
        elitism=neat_config.elitism,
        pair_selection_f=pair_selection_f,
    )

    if neat_config.do_speciation:
        species = speciate(
            next_gen_genotypes,
            compatibility_threshold=neat_config.compatibility_threshold,
            existing_species=next_gen_species
        )
    else:
        for gt in next_gen_genotypes:
            gt.species_id = 1

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario_id,
        generation=generation_n + 1,
        genotypes=next_gen_genotypes,
        fitness_f=fitness_f,
        pair_selection_f=pair_selection_f,
        neat_config=neat_config,
        ca_config=ca_config,
    )

    return '{scenario}, generation {generation_n}, {n_individuals} individuals, {n_species} species'.format(
        scenario=scenario,
        generation_n=generation_n,
        n_individuals=len(next_gen_genotypes),
        n_species=len(alive_species),
    )


@app.task(name='handle_individual', **AUTO_RETRY)
def handle_individual(scenario_id: int, generation: int, individual_number: int, genotype: Genome,
                      fitness_f: FITNESS_F_T, ca_config: CAConfig) -> Individual:
    λ = None
    phenotype = None

    if (genotype.fitness is None) or ca_config.compute_lambda:
        phenotype = create_feed_forward_phenotype(genotype)

    if genotype.fitness is None:
        phenotype = create_feed_forward_phenotype(genotype)

        try:
            genotype.fitness = fitness_f(phenotype, ca_config)
        except OverflowError:
            genotype.fitness = 0.0

        assert 0.0 <= genotype.fitness <= 1.0

    if ca_config.compute_lambda:
        try:
            λ = calculate_lambda(cppn=phenotype, ca_config=ca_config)
        except OverflowError:
            genotype.fitness = 0.0
            λ = 0.0

    individual = Individual(
        scenario_id=scenario_id,
        individual_number=individual_number,
        genotype=serialize_gt(genotype),
        fitness=genotype.fitness,
        generation=generation,
        λ=λ,
        species=UUID(int=genotype.species_id),
    )

    return individual
