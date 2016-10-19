import os
from datetime import datetime

from config import CAConfig, CPPNNEATConfig
from geometry.neighbourhoods import VON_NEUMANN
from run_experiment import initialize_scenario
from visualization.plot_fitness import plot_fitnesses_over_generations


def fitness_f(phenotype, ca_config: CAConfig):
    from geometry.cell_grid import FiniteCellGrid2D
    from ca.iterate import n_iterations

    neighbourhood = ca_config.neighbourhood
    alphabet = ca_config.alphabet

    r = 5

    initial = FiniteCellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        x_range=(-r, r),
        y_range=(-r, r),
        values={(0, 0): 1}
    )

    from utils import make_step_f
    step = make_step_f(0.5)

    def transition_f(args):
        t = tuple(int(x) for x in args)
        return step(phenotype.serial_activate(t)[0])

    grid = n_iterations(initial_grid=initial, transition_f=transition_f, n=10)

    return sum(sum(int(x) for x in row) for row in grid.get_whole()) / grid.area


if __name__ == '__main__':
    CA_CONFIG = CAConfig()
    CA_CONFIG.alphabet = (0, 1)
    CA_CONFIG.neighbourhood = VON_NEUMANN

    NEAT_CONFIG = CPPNNEATConfig()
    NEAT_CONFIG.generations = 100
    NEAT_CONFIG.pop_size = 100
    NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
    NEAT_CONFIG.weight_stdev = 1.0
    NEAT_CONFIG.compatibility_threshold = 0.5
    NEAT_CONFIG.prob_add_conn = 0.988
    NEAT_CONFIG.prob_add_node = 0.085
    NEAT_CONFIG.prob_delete_conn = 0.146
    NEAT_CONFIG.prob_delete_node = 0.0352
    NEAT_CONFIG.prob_mutate_bias = 0.0509
    NEAT_CONFIG.bias_mutation_power = 2.093
    NEAT_CONFIG.prob_mutate_response = 0.1
    NEAT_CONFIG.response_mutation_power = 0.1
    NEAT_CONFIG.prob_mutate_weight = 0.460
    NEAT_CONFIG.prob_replace_weight = 0.0245
    NEAT_CONFIG.weight_mutation_power = 0.825
    NEAT_CONFIG.prob_mutate_activation = 0.0
    NEAT_CONFIG.prob_toggle_link = 0.0138
    NEAT_CONFIG.max_weight = 30
    NEAT_CONFIG.min_weight = -30
    NEAT_CONFIG.excess_coefficient = 1.0
    NEAT_CONFIG.disjoint_coefficient = 1.0
    NEAT_CONFIG.weight_coefficient = 0.4
    NEAT_CONFIG.elitism = 2
    NEAT_CONFIG.stagnation_limit = 10
    NEAT_CONFIG.survival_threshold = 0.2

    DB_DIR = 'db/neat_test/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"Max one 2D CA"\npopulation size: {}\ngenerations: {}'.format(NEAT_CONFIG.pop_size,
                                                                                 NEAT_CONFIG.generations)
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=fitness_f,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )
    plot_fitnesses_over_generations(DB_PATH, title=DESCRIPTION, interval=10)
