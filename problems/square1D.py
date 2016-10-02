import os
from datetime import datetime

from geometry.cell_grid import CellGrid1D
from run_experiment import initialize_scenario
from selection import fitness_proportionate
from utils import random_string, splice, char_set


def fitness(phenotype):
    from utils import char_set
    from geometry.cell_grid import CellGrid1D
    INITIAL = CellGrid1D(cell_states=char_set(6))
    for x in range(5):
        INITIAL.set((x,), 'B')

    TARGET = CellGrid1D(cell_states=char_set(6))
    for x in range(25):
        TARGET.set((x,), char_set(6)[-1])

    transition_f = lambda k: phenotype[tuple(k)]

    from ca.iterate import iterate_until_stable
    grid = iterate_until_stable(initial_grid=INITIAL, transition_f=transition_f, max_n=100)

    score = 0.0
    for a, b in zip(grid.iterate_coords(), TARGET.iterate_coords()):
        if grid.get(a) == TARGET.get(b):
            score += 1.0 / 27

    return score


def geno_to_pheno_f(genotype):
    from ca.rule_tables import table_from_string
    from utils import char_set
    return table_from_string(genotype, char_set(6))


def mutation_f(genotype):
    from utils import char_set, mutate_char
    return mutate_char(genotype, char_set(6))


if __name__ == '__main__':
    grid = CellGrid1D(cell_states=(char_set(6)))
    POPULATION_SIZE = 10
    N_GENERATIONS = 100
    INITIAL_GENOTYPES = tuple(
        random_string(char_set(6), len(char_set(6)) ** len(grid.neighbourhood)) for _ in range(POPULATION_SIZE)
    )
    MUTATION_CHANCE = 0.01
    GENO_TO_PHENO_F = geno_to_pheno_f
    FITNESS_F = fitness
    SELECTION_F = fitness_proportionate
    CROSSOVER_F = splice
    MUTATION_F = mutation_f

    DB_DIR = 'db/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    description = '"Square 5", population size: {}, generations: {}'.format(POPULATION_SIZE, N_GENERATIONS)
    initialize_scenario(
        db_path=DB_PATH,
        description=description,
        initial_genotypes=INITIAL_GENOTYPES,
        population_size=POPULATION_SIZE,
        generations=N_GENERATIONS,
        geno_to_pheno_f=GENO_TO_PHENO_F,
        fitness_f=FITNESS_F,
        selection_f=SELECTION_F,
        crossover_f=CROSSOVER_F,
        mutation_f=MUTATION_F,
        mutation_chance=MUTATION_CHANCE,
    )
