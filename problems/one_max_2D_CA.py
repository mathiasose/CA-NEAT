import os
from datetime import datetime

from geometry.neighbourhoods import VON_NEUMANN
from plot import plot_fitnesses_over_generations
from run_experiment import initialize_scenario
from selection import sigma_scaled
from utils import random_string, splice


def fitness(phenotype, **kwargs):
    from geometry.cell_grid import CellGrid2D
    from ca.iterate import n_iterations

    neighbourhood = kwargs.get('neighbourhood')
    alphabet = kwargs.get('alphabet')

    initial = CellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        values=(((0, 0), '1'),)
    )

    transition_f = lambda k: phenotype[tuple(k)]
    grid = n_iterations(initial_grid=initial, transition_f=transition_f, n=10)

    return sum(sum(int(x) for x in row) for row in grid.get_rectangle((-5, 4), (-5, 4))) / 100


def geno_to_pheno_f(genotype, **kwargs):
    from ca.rule_tables import table_from_string
    return table_from_string(genotype, kwargs.get('alphabet'))


def mutation_f(genotype, **kwargs):
    from utils import mutate_char
    return mutate_char(genotype, kwargs.get('alphabet'))


if __name__ == '__main__':
    ALPHABET = '01'
    POPULATION_SIZE = 15
    N_GENERATIONS = 25
    NEIGHBOURHOOD = VON_NEUMANN
    INITIAL_GENOTYPES = tuple(
        random_string(ALPHABET, len(ALPHABET) ** len(NEIGHBOURHOOD)) for _ in range(POPULATION_SIZE)
    )
    MUTATION_CHANCE = 0.25
    GENO_TO_PHENO_F = geno_to_pheno_f
    FITNESS_F = fitness
    SELECTION_F = sigma_scaled
    CROSSOVER_F = splice
    MUTATION_F = mutation_f

    DB_DIR = 'db/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"Max one 2D CA"\npopulation size: {}\ngenerations: {}'.format(POPULATION_SIZE, N_GENERATIONS)
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        initial_genotypes=INITIAL_GENOTYPES,
        population_size=POPULATION_SIZE,
        generations=N_GENERATIONS,
        geno_to_pheno_f=GENO_TO_PHENO_F,
        fitness_f=FITNESS_F,
        selection_f=SELECTION_F,
        crossover_f=CROSSOVER_F,
        mutation_f=MUTATION_F,
        mutation_chance=MUTATION_CHANCE,
        alphabet=ALPHABET,
        neighbourhood=NEIGHBOURHOOD,
    )

    plot_fitnesses_over_generations(DB_PATH, title=DESCRIPTION, interval=1)
