import random
from random import choice
from statistics import median
from typing import Dict, List, Set, Tuple
from uuid import UUID

from celery.canvas import chain
from neat.genome import Genome
from neat.species import Species

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.database import Individual, Innovation, Scenario, get_db
from ca_neat.ga.population import create_initial_population, neat_reproduction, sort_into_species, speciate
from ca_neat.ga.selection import PAIR_SELECTION_F_T
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.run_experiment import AUTO_RETRY, FITNESS_F_T, handle_individual
from ca_neat.utils import pluck
from celery_app import app


def dummy_fitness_f(*args, **kwargs):
    raise Exception('This function should never be called')


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
        fitness_f=dummy_fitness_f,
        pair_selection_f=pair_selection_f,
        neat_config=neat_config,
        ca_config=ca_config,
        innovation_archive=[],
    )

    return scenario


def initialize_generation(db_path: str, scenario_id: int, generation: int, genotypes: List[Genome],
                          pair_selection_f: PAIR_SELECTION_F_T, fitness_f: FITNESS_F_T, neat_config: CPPNNEATConfig,
                          ca_config: CAConfig, innovation_archive: List[Genome]) -> None:
    from celery import group, chord

    distances = {}

    k = 15

    concurrent_tasks = []
    for i, gt in enumerate(genotypes):
        l = []
        for ot in genotypes + innovation_archive:
            if ot is gt:
                continue

            key = tuple(sorted((gt, ot), key=id))

            if key in distances:
                pass
            else:
                distances[key] = gt.distance(ot)

            l.append(distances[key])

        gt.fitness = sum(sorted(l)[:k]) / k

        concurrent_tasks.append(handle_individual.s(
            scenario_id=scenario_id,
            generation=generation,
            individual_number=i,
            genotype=gt,
            fitness_f=fitness_f,
            ca_config=ca_config,
        ))

    final_task = persist_results.subtask(
        args=(db_path, scenario_id, generation, fitness_f, pair_selection_f, neat_config, ca_config)
    )

    chord(group(concurrent_tasks), final_task)()


@app.task(name='persist_results_novelty', bind=True, **AUTO_RETRY)
def persist_results(task, results, db_path: str, scenario_id: int, generation_n: int, fitness_f: FITNESS_F_T,
                    pair_selection_f: PAIR_SELECTION_F_T, neat_config: CPPNNEATConfig, ca_config: CAConfig) -> str:
    db = get_db(db_path)
    session = db.Session()
    session.bulk_save_objects(results)

    added = 0
    for individual in results:
        if individual.fitness >= neat_config.innovation_threshold:
            session.add(Innovation(
                scenario_id=individual.scenario_id,
                generation=individual.generation,
                individual_number=individual.individual_number,
            ))
            added += 1

    if added == 0:
        individual = choice(results)
        session.add(Innovation(
            scenario_id=individual.scenario_id,
            generation=individual.generation,
            individual_number=individual.individual_number,
        ))

    session.commit()
    session.close()

    print('Generation {} added {} innovations'.format(generation_n, added))

    before = neat_config.innovation_threshold
    if added == 0:
        neat_config.innovation_threshold *= 0.95
    elif added > 5:
        neat_config.innovation_threshold *= 1.05

    if neat_config.innovation_threshold != before:
        print('Scenario {} adjusting innovation threshold from {:.2f} to {:.2f}'.format(scenario_id, before,
                                                                                        neat_config.innovation_threshold))
    check_if_done.delay(
        db_path=db_path,
        scenario_id=scenario_id,
        generation_n=generation_n,
        fitness_f=fitness_f,
        pair_selection_f=pair_selection_f,
        neat_config=neat_config,
        ca_config=ca_config
    )

    return 'Scenario {scenario_id}, generation {generation_n}, {n_individuals} individuals'.format(
        scenario_id=scenario_id,
        generation_n=generation_n,
        n_individuals=len(results),
    )


@app.task(name='check_if_done_novelty', bind=True, **AUTO_RETRY)
def check_if_done(task, db_path: str, scenario_id: int, generation_n: int, fitness_f: FITNESS_F_T,
                  pair_selection_f: PAIR_SELECTION_F_T, neat_config: CPPNNEATConfig, ca_config: CAConfig) -> str:
    db = get_db(db_path)
    session = db.Session()

    scenario = db.get_scenario(scenario_id, session=session)
    session.close()

    next_gen = generation_n + 1

    if next_gen >= scenario.generations:
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


@app.task(name='reproduction_io_novelty', bind=True, **AUTO_RETRY)
def reproduction_io(task, db_path: str, scenario_id: int, generation_n: int, neat_config: CPPNNEATConfig
                    ) -> Tuple[Scenario, Set[Species]]:
    db = get_db(db_path)
    session = db.Session()

    scenario = db.get_scenario(scenario_id, session=session)
    population = db.get_generation(scenario_id, generation_n, session=session)

    species = sort_into_species(
        genotypes=(
            deserialize_gt(
                gt_json_bytes=individual.genotype,
                neat_config=neat_config
            ) for individual in population
        )
    )

    stagnation_limit = neat_config.stagnation_limit
    if (not neat_config.do_speciation) or (not stagnation_limit) or (generation_n < stagnation_limit):
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
                    .filter(Individual.species == UUID(int=spec.ID)) \
                    .order_by(Individual.generation)
            )
            m = median(pluck(individuals, 'fitness'))
            medians[spec.ID] = m
            stagnant[spec.ID] = individuals[0].fitness == individuals[-1].fitness

        median_of_medians = median(medians.values())

        alive_species = set(s for s in species if (not stagnant[s.ID]) or (medians[s.ID] >= median_of_medians))

    innovation_archive = [deserialize_gt(x[0], neat_config) for x in db.get_innovations(scenario_id, session=session)]
    session.close()

    return scenario, alive_species, innovation_archive


@app.task(name='reproduction_novelty', bind=True, **AUTO_RETRY)
def reproduction(task, results, db_path: str, scenario_id: int, generation_n: int, fitness_f: FITNESS_F_T,
                 pair_selection_f: PAIR_SELECTION_F_T, neat_config: CPPNNEATConfig, ca_config: CAConfig) -> str:
    scenario, alive_species, innovation_archive = results

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
        innovation_archive=innovation_archive,
    )

    return '{scenario}, generation {generation_n}, {n_individuals} individuals, {n_species} species'.format(
        scenario=scenario,
        generation_n=generation_n,
        n_individuals=len(next_gen_genotypes),
        n_species=len(alive_species),
    )
