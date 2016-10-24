import logging
from typing import List

from celery.app import shared_task
from neat.genome import Genome
from neat.nn import create_feed_forward_phenotype
from neat.population import CompleteExtinctionException
from sqlalchemy.sql.functions import now

from add_dill import add_dill
from config import CAConfig, CPPNNEATConfig
from database import Db, Individual, Scenario
from run_neat import (create_initial_population, neat_reproduction,
                      sort_into_species, speciate)
from stagnation import (get_total_fitnesses_by_species_by_generation,
                        is_species_stagnant)

WAIT = 5

add_dill()


def get_db(path):
    return Db(path, echo=False)


def initialize_scenario(db_path: str, description: str, fitness_f, neat_config: CPPNNEATConfig, ca_config: CAConfig):
    scenario = Scenario(description=description, generations=neat_config.generations,
                        population_size=neat_config.pop_size)
    get_db(db_path).save_scenario(scenario)

    initial_genotypes = list(create_initial_population(neat_config))

    species = speciate(initial_genotypes, compatibility_threshold=neat_config.compatibility_threshold)

    assert len(species) > 1

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario.id,
        generation=0,
        genotypes=initial_genotypes,
        fitness_f=fitness_f,
        neat_config=neat_config,
        ca_config=ca_config,
    )


def initialize_generation(db_path: str, scenario_id: int, generation: int, genotypes: List[Genome],
                          fitness_f, neat_config: CPPNNEATConfig, ca_config: CAConfig):
    from celery import group, chord

    grouped_tasks = group(handle_individual.s(
        db_path=db_path,
        scenario_id=scenario_id,
        generation=generation,
        individual_number=i,
        genotype=genotype,
        fitness_f=fitness_f,
        neat_config=neat_config,
        ca_config=ca_config,
    ) for i, genotype in enumerate(genotypes))

    final_task = finalize_generation.subtask(args=(db_path, scenario_id, generation, fitness_f, neat_config, ca_config))

    chord(grouped_tasks, final_task)()


@shared_task(name='finalize_generation', bind=True)
def finalize_generation(task, results, db_path: str, scenario_id: int, generation: int, fitness_f,
                        neat_config: CPPNNEATConfig, ca_config: CAConfig):
    """
    'results' is return values of preceding group of tasks, can safely be ignored
    """
    db = get_db(db_path)
    scenario = db.get_scenario(scenario_id)
    population = db.get_generation(scenario_id, generation)

    assert population.count() == scenario.population_size

    next_gen = generation + 1

    if next_gen == scenario.generations:
        logging.info('Scenario {} finished after {} generations'.format(scenario_id, scenario.generations))
        return

    species = sort_into_species([individual.genotype for individual in population])

    if (not neat_config.stagnation_limit) or (generation < neat_config.stagnation_limit):
        alive_species = species
    else:
        fitnesses = get_total_fitnesses_by_species_by_generation(db)
        alive_species = [s for s in species if not is_species_stagnant(fitnesses, s.ID, neat_config.stagnation_limit)]

    if not alive_species:
        raise CompleteExtinctionException

    new_species, new_genotypes = neat_reproduction(
        species=alive_species, pop_size=scenario.population_size, survival_threshold=neat_config.survival_threshold,
        elitism=neat_config.elitism)

    assert len(new_genotypes) == scenario.population_size

    speciate(new_genotypes, compatibility_threshold=neat_config.compatibility_threshold, existing_species=new_species)

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario_id,
        generation=next_gen,
        genotypes=new_genotypes,
        fitness_f=fitness_f,
        neat_config=neat_config,
        ca_config=ca_config,
    )

    logging.info('Finished generation {} of scenario {}'.format(generation, scenario_id))
    return generation


@shared_task(name='handle_individual')
def handle_individual(db_path: str, scenario_id: int, generation: int, individual_number: int, genotype: Genome,
                      fitness_f, neat_config: CPPNNEATConfig, ca_config: CAConfig):
    phenotype = create_feed_forward_phenotype(genotype)
    fitness = fitness_f(phenotype=phenotype, ca_config=ca_config)

    assert 0.0 <= fitness <= 1.0

    if hasattr(genotype, 'fitness'):
        genotype.fitness = fitness

    individual = Individual(
        scenario_id=scenario_id,
        individual_number=individual_number,
        genotype=genotype,
        fitness=fitness,
        generation=generation,
        timestamp=now()
    )

    get_db(db_path).save_individual(individual)

    return individual.genotype
