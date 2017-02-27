import logging
from statistics import median
from typing import Callable, Dict, List
from uuid import UUID

from neat.genome import Genome
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype
from neat.population import CompleteExtinctionException
from sqlalchemy.exc import OperationalError

from ca_neat.ca.calculate_lambda import calculate_lambda
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
    'autoretry_for': (OperationalError,),
    'retry_kwargs': {'countdown': 5},
}


def initialize_scenario(db_path: str, description: str, fitness_f: FITNESS_F_T, pair_selection_f: PAIR_SELECTION_F_T,
                        neat_config: CPPNNEATConfig, ca_config: CAConfig) -> Scenario:
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

    initial_genotypes = list(create_initial_population(neat_config))
    speciate(initial_genotypes, compatibility_threshold=neat_config.compatibility_threshold)

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

    final_task = finalize_generation.subtask(
        args=(db_path, scenario_id, generation, fitness_f, pair_selection_f, neat_config, ca_config)
    )

    chord(grouped_tasks, final_task)()


@app.task(name='finalize_generation', bind=True, **AUTO_RETRY)
def finalize_generation(task, results, db_path: str, scenario_id: int, generation_n: int, fitness_f: FITNESS_F_T,
                        pair_selection_f: PAIR_SELECTION_F_T, neat_config: CPPNNEATConfig, ca_config: CAConfig) -> str:
    db = get_db(db_path)
    session = db.Session()
    session.bulk_save_objects(results)
    session.commit()

    scenario = db.get_scenario(scenario_id, session=session)
    population = db.get_generation(scenario_id, generation_n, session=session)

    next_gen = generation_n + 1

    optimal_solution = population.filter(Individual.fitness >= 1.0)
    optimal_found = (neat_config.stop_when_optimal_found and session.query(optimal_solution.exists()).scalar())

    if (next_gen >= scenario.generations) or optimal_found:
        msg = 'Scenario {} finished after {} generations'.format(scenario_id, next_gen)
        send_message_via_pushbullet.delay(title=db_path, body=msg)
        logging.info(msg)
        return msg

    species = sort_into_species(
        genotypes=(
            deserialize_gt(
                gt_json_bytes=individual.genotype,
                neat_config=neat_config
            ) for individual in population
        )
    )

    stagnation_limit = neat_config.stagnation_limit
    if (not stagnation_limit) or (generation_n < stagnation_limit):
        alive_species = species
    else:
        all_individuals = db \
            .get_individuals(scenario_id=scenario_id, session=session) \
            .filter(generation_n >= generation_n - stagnation_limit)

        stagnant: Dict[int, bool] = {}
        medians: Dict[int, float] = {}
        for spec in species:
            individuals = list(
                all_individuals \
                    .filter(Individual.species == spec.ID) \
                    .order_by(Individual.generation)
            )
            m = median(pluck(individuals, 'fitness'))
            medians[spec.ID] = m
            stagnant[spec.ID] = individuals[0].fitness == individuals[-1].fitness

        median_of_medians = median(medians.values())

        alive_species = set(s for s in species if (not stagnant[s.ID]) or (medians[s.ID] >= median_of_medians))

    if not alive_species:
        send_message_via_pushbullet(
            title='Halted: {}'.format(scenario.description),
            body='Complete extinction for scenario {} at generation {}'.format(
                scenario_id,
                generation_n,
            ),
        )
        raise CompleteExtinctionException

    next_gen_species, nex_gen_genotypes = neat_reproduction(
        species=alive_species,
        pop_size=scenario.population_size,
        survival_threshold=neat_config.survival_threshold,
        elitism=neat_config.elitism,
        pair_selection_f=pair_selection_f,
    )

    species = speciate(
        nex_gen_genotypes,
        compatibility_threshold=neat_config.compatibility_threshold,
        existing_species=next_gen_species
    )

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario_id,
        generation=next_gen,
        genotypes=nex_gen_genotypes,
        fitness_f=fitness_f,
        pair_selection_f=pair_selection_f,
        neat_config=neat_config,
        ca_config=ca_config,
    )

    session.close()

    return '{scenario}, generation {generation_n}, {n_individuals} individuals, {n_species} species'.format(
        scenario=scenario,
        generation_n=generation_n,
        n_individuals=len(nex_gen_genotypes),
        n_species=len(alive_species),
    )


@app.task(name='handle_individual', **AUTO_RETRY)
def handle_individual(scenario_id: int, generation: int, individual_number: int, genotype: Genome,
                      fitness_f: FITNESS_F_T, ca_config: CAConfig) -> Individual:
    phenotype = create_feed_forward_phenotype(genotype)
    try:
        genotype.fitness = fitness_f(phenotype, ca_config)
    except OverflowError:
        genotype.fitness = 0.0

    assert 0.0 <= genotype.fitness <= 1.0

    if ca_config.compute_lambda:
        λ = calculate_lambda(cppn=phenotype, ca_config=ca_config)
    else:
        λ = None

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
