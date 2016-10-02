import logging
from random import random

from celery.app import shared_task
from sqlalchemy.sql.functions import now

from add_dill import add_dill
from database import Db, Individual, Scenario

add_dill()


def get_db(path):
    return Db(path, echo=False)


def initialize_scenario(db_path, initial_genotypes, description, generations, population_size, **kwargs):
    scenario = Scenario(description=description, generations=generations, population_size=population_size)
    get_db(db_path).save_scenario(scenario)

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
        **kwargs
    )


def initialize_generation(genotypes, **kwargs):
    for i, genotype in enumerate(genotypes):
        handle_individual.delay(
            individual_number=i,
            genotype=genotype,
            **kwargs
        )


@shared_task(name='finalize_generation', bind=True)
def finalize_generation(task, db_path, scenario_id, generation, selection_f, crossover_f, mutation_f, mutation_chance,
                        **kwargs):
    db = get_db(db_path)
    scenario = db.get_scenario(scenario_id)
    population_size_target = scenario.population_size
    population = db.get_generation(scenario_id, generation)

    try:
        assert population.count() == population_size_target
    except AssertionError:
        raise task.retry(countdown=5)

    next_gen = generation + 1

    if next_gen == scenario.generations:
        logging.info('Scenario {} finished after {} generations'.format(scenario_id, scenario.generations))
        return

    new_genotypes = []
    pair_generator = selection_f(population=population)
    while len(new_genotypes) < population_size_target:
        a, b = next(pair_generator)
        genotype = crossover_f(a=a.genotype, b=b.genotype)

        if random() < mutation_chance:
            genotype = mutation_f(genotype=genotype)

        assert len(genotype) == len(a.genotype)

        new_genotypes.append(genotype)

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
        **kwargs
    )

    logging.info('Finished generation {} of scenario {}'.format(generation, scenario_id))


@shared_task(name='handle_individual')
def handle_individual(db_path, scenario_id, generation, individual_number, genotype, geno_to_pheno_f, fitness_f,
                      **kwargs):
    phenotype = geno_to_pheno_f(genotype=genotype)
    fitness = fitness_f(phenotype=phenotype)

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
