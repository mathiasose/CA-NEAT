import os
from datetime import datetime

from config import CAConfig, CPPNNEATConfig
from geometry.neighbourhoods import VON_NEUMANN
from run_experiment import initialize_scenario
from visualization.plot_fitness import plot_fitnesses_over_generations


def fitness_f(phenotype, ca_config: CAConfig):
    from utils import is_even
    from ca.iterate import ca_develop
    from math import exp

    r = ca_config.r

    grid = ca_develop(phenotype, ca_config)

    correct_count = 0
    for y in range(-r, r):
        for x in range(-r, r):
            v = grid.get((x, y))
            if is_even(x) == is_even(y) and v == 1:
                correct_count += 1
            elif is_even(x) != is_even(y) and v == 0:
                correct_count += 1

    x = correct_count / grid.area

    k = 5
    redistribute = lambda x: exp(k * x) / exp(k)

    return redistribute(x)


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = (0, 1)
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.r = 5
CA_CONFIG.iterations = 10

NEAT_CONFIG = CPPNNEATConfig()
NEAT_CONFIG.pop_size = 100
NEAT_CONFIG.generations = 100
NEAT_CONFIG.stagnation_limit = 10
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
NEAT_CONFIG.survival_threshold = 0.2

if __name__ == '__main__':
    DB_DIR = 'db/checkerboard/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"10x10 Checkerboard CA"\npopulation size: {}\ngenerations: {}'.format(
        NEAT_CONFIG.pop_size,
        NEAT_CONFIG.generations
    )
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=fitness_f,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )
    plot_fitnesses_over_generations(DB_PATH, title=DESCRIPTION, interval=10)
