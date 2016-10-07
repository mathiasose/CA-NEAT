import os
from datetime import datetime

from geometry.neighbourhoods import VON_NEUMANN
from run_experiment import initialize_scenario
from selection import sigma_scaled
from utils import char_set, random_string, splice


def fitness(phenotype, **kwargs):
    from geometry.cell_grid import CellGrid1D
    from ca.iterate import iterate_until_stable

    neighbourhood = kwargs.get('neighbourhood')
    alphabet = kwargs.get('alphabet')

    initial = CellGrid1D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        values=(((x,), 'B') for x in range(5))
    )
    target = CellGrid1D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        values=(((x,), alphabet[-1]) for x in range(25))
    )

    transition_f = lambda k: phenotype[tuple(k)]
    grid = iterate_until_stable(initial_grid=initial, transition_f=transition_f, max_n=100)

    # TODO make proper calculation
    score = 0.0
    for a, b in zip(grid.iterate_coords(), target.iterate_coords()):
        if grid.get(a) == target.get(b):
            score += 1.0 / 27

    return score


def geno_to_pheno_f(genotype, **kwargs):
    from ca.rule_tables import table_from_string
    return table_from_string(genotype, kwargs.get('alphabet'))


def mutation_f(genotype, **kwargs):
    from utils import mutate_char
    return mutate_char(genotype, kwargs.get('alphabet'))


if __name__ == '__main__':
    ALPHABET = char_set(8)
    POPULATION_SIZE = 10
    N_GENERATIONS = 1000
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

    DESCRIPTION = '"Square 5", population size: {}, generations: {}'.format(POPULATION_SIZE, N_GENERATIONS)
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
