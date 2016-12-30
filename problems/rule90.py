from datetime import datetime

import os

from ga.selection import fitness_proportionate
from run_experiment import initialize_scenario
from utils import mutate_bit, random_bitstring, splice


def geno_to_pheno(genotype):
    return {bin(i).lstrip('0b').rjust(3, '0'): b for i, b in enumerate(genotype)}


def fitness(phenotype):
    def develop(world):
        world = '0' + world + '0'
        for i, b in enumerate(world):
            if i == 0:
                yield '0' + world[:2]
            elif i == len(world) - 1:
                yield world[-2:] + '0'
            else:
                yield world[i - 1:i + 2]

    rule90 = {bin(i).lstrip('0b').rjust(3, '0'): b for i, b in enumerate('01011010')}
    world = benchmark = '010'

    for _ in range(15):
        benchmark = ''.join(map(rule90.get, develop(benchmark)))
        world = ''.join(map(phenotype.get, develop(world)))

    return sum(a == b for a, b in zip(world, benchmark)) / len(benchmark)


if __name__ == '__main__':
    POPULATION_SIZE = 10
    INITIAL_GENOTYPES = tuple(random_bitstring(8) for _ in range(POPULATION_SIZE))
    MUTATION_CHANCE = 0.01
    GENO_TO_PHENO_F = geno_to_pheno
    FITNESS_F = fitness
    SELECTION_F = fitness_proportionate
    CROSSOVER_F = splice
    MUTATION_F = mutate_bit

    DB_DIR = 'db/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    for n_generations in (100,):
        description = '"Rule 90 1D CA", population size: {}, generations: {}'.format(POPULATION_SIZE, n_generations)
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
