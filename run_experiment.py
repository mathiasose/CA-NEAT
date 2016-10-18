import logging
from typing import List

from celery.app import shared_task
from neat.genome import Genome
from neat.population import CompleteExtinctionException
from sqlalchemy.sql.functions import now

from add_dill import add_dill
from database import Db, Individual, Scenario
from run_neat import neat_reproduction, sort_into_species, speciate
from stagnation import (get_total_fitnesses_by_species_by_generation,
                        is_species_stagnant)

RETRY = 5

add_dill()


def get_db(path):
    return Db(path, echo=False)


def initialize_scenario(db_path: str, initial_genotypes: List[Genome], description: str, generations: int,
                        population_size: int, compatibility_threshold: float, **kwargs):
    scenario = Scenario(description=description, generations=generations, population_size=population_size)
    get_db(db_path).save_scenario(scenario)

    speciate(initial_genotypes, compatibility_threshold=compatibility_threshold)

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario.id,
        generation=0,
        genotypes=initial_genotypes,
        **kwargs
    )

    finalize_generation.delay(
        db_path=db_path,
        scenario_id=scenario.id,
        generation=0,
        compatibility_threshold=compatibility_threshold,
        **kwargs
    )


def initialize_generation(genotypes: List[Genome], **kwargs):
    for i, genotype in enumerate(genotypes):
        handle_individual.delay(
            individual_number=i,
            genotype=genotype,
            **kwargs
        )


@shared_task(name='finalize_generation', bind=True)
def finalize_generation(task, db_path: str, scenario_id: int, generation: int, selection_f,
                        crossover_f, mutation_f, mutation_chance: float, stagnation_limit: int,
                        survival_threshold: float, compatibility_threshold: float, elitism=0, **kwargs):
    db = get_db(db_path)
    scenario = db.get_scenario(scenario_id)
    population_size_target = scenario.population_size
    population = db.get_generation(scenario_id, generation)

    assert population.count() <= population_size_target  # something is terribly wrong if this fails

    try:
        assert population.count() == population_size_target
    except AssertionError:
        raise task.retry(countdown=RETRY)

    next_gen = generation + 1

    if next_gen == scenario.generations:
        logging.info('Scenario {} finished after {} generations'.format(scenario_id, scenario.generations))
        return

    species = sort_into_species([individual.genotype for individual in population])

    if (not stagnation_limit) or (generation < stagnation_limit):
        alive_species = species
    else:
        total_fitnesses = get_total_fitnesses_by_species_by_generation(db)
        alive_species = [s for s in species if not is_species_stagnant(total_fitnesses, s.ID, stagnation_limit)]

    if not alive_species:
        raise CompleteExtinctionException

    new_species, new_genotypes = neat_reproduction(species=alive_species, pop_size=scenario.population_size,
                                                   survival_threshold=survival_threshold, elitism=elitism)
    speciate(new_genotypes, compatibility_threshold=compatibility_threshold, existing_species=new_species)

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario_id,
        generation=next_gen,
        genotypes=new_genotypes,
        **kwargs
    )

    finalize_generation.delay(
        db_path=db_path,
        scenario_id=scenario_id,
        generation=next_gen,
        selection_f=selection_f,
        crossover_f=crossover_f,
        mutation_f=mutation_f,
        mutation_chance=mutation_chance,
        stagnation_limit=stagnation_limit,
        survival_threshold=survival_threshold,
        compatibility_threshold=compatibility_threshold,
        elitism=elitism,
        **kwargs
    )

    logging.info('Finished generation {} of scenario {}'.format(generation, scenario_id))
    return generation


@shared_task(name='handle_individual')
def handle_individual(db_path: str, scenario_id: int, generation: int, individual_number: int, genotype: Genome,
                      geno_to_pheno_f, fitness_f, **kwargs):
    phenotype = geno_to_pheno_f(genotype=genotype, **kwargs)
    fitness = fitness_f(phenotype=phenotype, **kwargs)

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
