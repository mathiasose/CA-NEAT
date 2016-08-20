import logging
import os
from datetime import datetime
from random import random

from celery.app import shared_task
from sqlalchemy.sql.functions import now

from add_dill import add_dill
from database import Db, Individual, Scenario
from selection import fitness_proportionate
from utils import mutate, random_bitstring, splice

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
def finalize_generation(self, db_path, scenario_id, generation, selection_f, crossover_f, mutation_f, mutation_chance,
                        **kwargs):
    db = get_db(db_path)
    scenario = db.get_scenario(scenario_id)
    population_size_target = scenario.population_size
    population = db.get_generation(scenario_id, generation)

    try:
        assert population.count() == population_size_target
    except AssertionError:
        raise self.retry(countdown=5)

    next_gen = generation + 1

    if next_gen == scenario.generations:
        logging.info('Scenario {} finished after {} generations'.format(scenario_id, scenario.generations))
        return

    pairs = selection_f(population=population, n_pairs=population_size_target)
    new_genotypes = tuple(crossover_f(a.genotype, b.genotype) for (a, b) in pairs)
    new_genotypes = tuple(mutation_f(x) if random() < mutation_chance else x for x in new_genotypes)

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
    phenotype = geno_to_pheno_f(genotype)
    fitness = fitness_f(phenotype)

    individual = Individual(
        scenario_id=scenario_id,
        individual_number=individual_number,
        genotype=genotype,
        fitness=fitness,
        generation=generation,
        timestamp=now()
    )

    get_db(db_path).save_individual(individual)


if __name__ == '__main__':
    POPULATION_SIZE = 10
    INITIAL_GENOTYPES = tuple(random_bitstring(16) for _ in range(POPULATION_SIZE))
    MUTATION_CHANCE = 0.1
    GENO_TO_PHENO_F = lambda x: x
    FITNESS_F = lambda bitstr: sum(c == '1' for c in bitstr)
    SELECTION_F = fitness_proportionate
    CROSSOVER_F = splice
    MUTATION_F = mutate

    DB_DIR = 'db/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    for n_generations in (10, 50, 100):
        description = '"Max ones", population size: {}, generations: {}'.format(POPULATION_SIZE, n_generations)
        initialize_scenario(
            db_path=DB_PATH,
            description=description,
            initial_genotypes=INITIAL_GENOTYPES,
            population_size=POPULATION_SIZE,
            generations=n_generations,
            geno_to_pheno_f=GENO_TO_PHENO_F,
            fitness_f=FITNESS_F,
            selection_f=SELECTION_F,
            crossover_f=CROSSOVER_F,
            mutation_f=MUTATION_F,
            mutation_chance=MUTATION_CHANCE,
        )
