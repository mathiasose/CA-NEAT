import os
from datetime import datetime

from run_experiment import initialize_scenario
from selection import fitness_proportionate
from utils import random_bitstring, splice, mutate_bit

if __name__ == '__main__':
    POPULATION_SIZE = 10
    INITIAL_GENOTYPES = tuple(random_bitstring(16) for _ in range(POPULATION_SIZE))
    MUTATION_CHANCE = 0.1
    GENO_TO_PHENO_F = lambda genotype: genotype
    FITNESS_F = lambda phenotype: sum(c == '1' for c in phenotype)
    SELECTION_F = fitness_proportionate
    CROSSOVER_F = splice
    MUTATION_F = mutate_bit

    DB_DIR = 'db/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    for n_generations in (10,):
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
