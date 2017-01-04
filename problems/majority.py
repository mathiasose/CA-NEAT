import os
from datetime import datetime
from statistics import mode

from config import CAConfig, CPPNNEATConfig
from ga.selection import sigma_scaled
from geometry.neighbourhoods import LLLCRRR
from patterns.patterns import ALPHABET_2
from run_experiment import initialize_scenario


def create_binary_pattern(alphabet):
    from random import choice, shuffle

    a, b = alphabet

    xa, xb = (0, 149)
    r = xb - xa

    values = [a] * (r // 3) + [b] * (r // 3)

    while len(values) < r:
        values.append(choice(alphabet))

    shuffle(values)

    return values


def invert_pattern(pattern):
    a, b = ALPHABET_2

    return list(map(lambda value: (a if value == b else b), pattern))


def fitness_f(phenotype, ca_config: CAConfig) -> float:
    from ca.iterate import iterate_ca_n_times_or_until_cycle_found
    from utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from typing import Tuple, T
    from statistics import mean
    from geometry.cell_grid import ToroidalCellGrid1D
    from math import exp

    alphabet = ca_config.alphabet

    iterations = ca_config.iterations
    state_normalization_rules = create_state_normalization_rules(states=alphabet)

    def ca_develop(grid, network: FeedForwardNetwork):
        def transition_f(inputs_discrete_values: Tuple[T]) -> T:
            if all((x == grid.dead_cell) for x in inputs_discrete_values):
                return grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield grid

        for grid_it in iterate_ca_n_times_or_until_cycle_found(grid, transition_f, iterations):
            yield grid_it

    k = 1
    redistribute = lambda x: x * exp(k * x) / exp(k)

    results = []
    for pattern, majority in ca_config.etc['test_patterns']:
        x_range = (0, len(pattern))
        neighbourhood = ca_config.neighbourhood

        init_grid = ToroidalCellGrid1D(
            cell_states=alphabet,
            x_range=x_range,
            values={(i,): x for i, x in enumerate(pattern)},
            neighbourhood=neighbourhood,
        )

        grid_iterations = tuple(ca_develop(init_grid, phenotype))
        values = grid_iterations[-1].get_whole()

        results.append(
            sum(x == majority for x in values) / init_grid.area
        )

    return redistribute(mean(results))


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_2
CA_CONFIG.neighbourhood = LLLCRRR
CA_CONFIG.iterations = 149

patterns = [create_binary_pattern(alphabet=CA_CONFIG.alphabet) for _ in range(1)]
patterns.extend(list(map(invert_pattern, patterns)))
CA_CONFIG.etc = {
    'test_patterns': [(pattern, mode(pattern)) for pattern in patterns]
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 20
NEAT_CONFIG.generations = 10000
NEAT_CONFIG.elitism = 1
NEAT_CONFIG.survival_threshold = 0.2
NEAT_CONFIG.stagnation_limit = 15

NEAT_CONFIG.compatibility_threshold = 3.0
NEAT_CONFIG.excess_coefficient = 1.0
NEAT_CONFIG.disjoint_coefficient = 1.0
NEAT_CONFIG.weight_coefficient = 0.5

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

NEAT_CONFIG.max_weight = 30
NEAT_CONFIG.min_weight = -30
NEAT_CONFIG.weight_stdev = 1.0

NEAT_CONFIG.prob_add_conn = 0.5
NEAT_CONFIG.prob_add_node = 0.5
NEAT_CONFIG.prob_delete_conn = 0.25
NEAT_CONFIG.prob_delete_node = 0.25
NEAT_CONFIG.prob_mutate_bias = 0.8
NEAT_CONFIG.bias_mutation_power = 0.5
NEAT_CONFIG.prob_mutate_response = 0.8
NEAT_CONFIG.response_mutation_power = 0.5
NEAT_CONFIG.prob_mutate_weight = 0.8
NEAT_CONFIG.prob_replace_weight = 0.1
NEAT_CONFIG.weight_mutation_power = 0.5
NEAT_CONFIG.prob_mutate_activation = 0.002
NEAT_CONFIG.prob_toggle_link = 0.01

PAIR_SELECTION_F = sigma_scaled

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'results', PROBLEM_NAME))
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    DB_PATH = 'sqlite:///' + os.path.join(RESULTS_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"Majority problem"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(1):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=fitness_f,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
