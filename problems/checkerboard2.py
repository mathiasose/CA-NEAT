from datetime import datetime

import os

from config import CAConfig, CPPNNEATConfig
from geometry.neighbourhoods import VON_NEUMANN, EXTENDED_VON_NEUMANN
from run_experiment import initialize_scenario
from selection import sigma_scaled
from visualization.plot_fitness import plot_fitnesses_over_generations


def fitness_f(phenotype, ca_config: CAConfig):
    from ca.iterate import ca_develop
    from patterns.checkerboard import evaluate
    from math import exp
    from geometry.cell_grid import FiniteCellGrid2D

    grid_cell = ca_config.etc['checkerboard_grid_cell']

    neighbourhood = ca_config.neighbourhood
    alphabet = ca_config.alphabet
    r = ca_config.r

    initial_grid = FiniteCellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        x_range=(-r, r),
        y_range=(-r, r),
        values={(0, 0): '1'},
    )

    grid_iterations = ca_develop(phenotype, ca_config, initial_grid)
    x = evaluate(grid_iterations[-1], grid_cell=grid_cell, grid_r=r)

    k = 5
    redistribute = lambda x: x * exp(k * x) / exp(k)

    return redistribute(x)


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ('0', '1')
CA_CONFIG.neighbourhood = EXTENDED_VON_NEUMANN
CA_CONFIG.r = 10
CA_CONFIG.iterations = 50
CA_CONFIG.etc = {
    'checkerboard_grid_cell': 2,
}

NEAT_CONFIG = CPPNNEATConfig()
NEAT_CONFIG.pop_size = 100
NEAT_CONFIG.generations = 100
NEAT_CONFIG.stagnation_limit = 5
NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.weight_stdev = 1.0
NEAT_CONFIG.compatibility_threshold = 0.5
NEAT_CONFIG.prob_add_conn = 0.988
NEAT_CONFIG.prob_add_node = 0.185
NEAT_CONFIG.prob_delete_conn = 0.146
NEAT_CONFIG.prob_delete_node = 0.0352
NEAT_CONFIG.prob_mutate_bias = 0.0509
NEAT_CONFIG.bias_mutation_power = 2.093
NEAT_CONFIG.prob_mutate_response = 0.2
NEAT_CONFIG.response_mutation_power = 0.1
NEAT_CONFIG.prob_mutate_weight = 0.460
NEAT_CONFIG.prob_replace_weight = 0.0745
NEAT_CONFIG.weight_mutation_power = 0.825
NEAT_CONFIG.prob_mutate_activation = 0.0
NEAT_CONFIG.prob_toggle_link = 0.0138
NEAT_CONFIG.max_weight = 30
NEAT_CONFIG.min_weight = -30
NEAT_CONFIG.excess_coefficient = 1.0
NEAT_CONFIG.disjoint_coefficient = 1.0
NEAT_CONFIG.weight_coefficient = 0.4
NEAT_CONFIG.elitism = 1
NEAT_CONFIG.survival_threshold = 0.5

if __name__ == '__main__':
    DB_DIR = 'db/2_checkerboard/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"{d}x{d} {n}-Checkerboard CA"\npopulation size: {pop}\ngenerations: {gens}'.format(
        d=2 * CA_CONFIG.r,
        n=CA_CONFIG.etc['checkerboard_grid_cell'],
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=fitness_f,
        pair_selection_f=sigma_scaled,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )
    plot_fitnesses_over_generations(DB_PATH, title=DESCRIPTION, interval=10)