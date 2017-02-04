import os
from datetime import datetime

from neat.nn import FeedForwardNetwork

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import VON_NEUMANN
from ca_neat.patterns.patterns import ALPHABET_4, NORWEGIAN, SEED_7X7
from ca_neat.run_experiment import initialize_scenario

CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_4
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 30
CA_CONFIG.etc = {
    'target_pattern': NORWEGIAN,
    'seed': SEED_7X7,
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 700
NEAT_CONFIG.elitism = 1
NEAT_CONFIG.survival_threshold = 0.2
NEAT_CONFIG.stagnation_limit = 15

NEAT_CONFIG.compatibility_threshold = 3.0
NEAT_CONFIG.excess_coefficient = 1.0
NEAT_CONFIG.disjoint_coefficient = 1.0
NEAT_CONFIG.weight_coefficient = 0.5

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood) + 2
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


def morphogenesis_fitness_f_with_coord_input(phenotype: FeedForwardNetwork, ca_config: CAConfig) -> float:
    from ca_neat.ca.iterate import iterate_ca_once_with_coord_inputs, iterate_ca_n_times_or_until_cycle_found
    from ca_neat.patterns.replicate_pattern import count_correct_cells
    from ca_neat.geometry.cell_grid import ToroidalCellGrid2D
    from ca_neat.utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from math import exp
    from typing import Sequence, Iterator
    from ca_neat.geometry.cell_grid import CELL_STATE_T
    from ca_neat.geometry.neighbourhoods import radius_2d

    neighbourhood = ca_config.neighbourhood
    alphabet = ca_config.alphabet
    target_pattern = ca_config.etc['target_pattern']
    seed = ca_config.etc['seed']
    iterations = ca_config.iterations

    pattern_w = len(target_pattern[0])
    pattern_h = len(target_pattern)

    initial_grid = ToroidalCellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        x_range=(0, pattern_w),
        y_range=(0, pattern_h),
    )
    initial_grid.add_pattern_at_coord(seed, (0, 0))

    state_normalization_rules = create_state_normalization_rules(states=alphabet)
    r = radius_2d(neighbourhood)
    coord_normalization_rules = create_state_normalization_rules(states=range(0 - r, max(pattern_h, pattern_w) + r + 1))

    def ca_develop(network: FeedForwardNetwork) -> Iterator[ToroidalCellGrid2D]:
        def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
            neighbour_values, xy_values = inputs_discrete_values[:-2], inputs_discrete_values[-2:]

            if all((x == initial_grid.dead_cell) for x in neighbour_values):
                return initial_grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[n] for n in neighbour_values) + \
                                  tuple(coord_normalization_rules[n] for n in xy_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield initial_grid

        for grid in iterate_ca_n_times_or_until_cycle_found(
                initial_grid=initial_grid,
                transition_f=transition_f,
                n=iterations,
                iterate_f=iterate_ca_once_with_coord_inputs
        ):
            yield grid

    grid_iterations = ca_develop(phenotype)

    best = 0.0
    for i, grid in enumerate(grid_iterations):
        correctness_fraction = count_correct_cells(grid.get_whole(), target_pattern=target_pattern) / grid.area

        if correctness_fraction >= 1.0:
            return correctness_fraction

        if correctness_fraction > best:
            best = correctness_fraction

    k = 5
    redistribute = lambda x: x * exp(k * x) / exp(k)

    return redistribute(best)


FITNESS_F = morphogenesis_fitness_f_with_coord_input

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'results', PROBLEM_NAME))
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    DB_PATH = 'sqlite:///' + os.path.join(RESULTS_DIR, '{}.db'.format('2017-01-30 09:30:09.020771'))
    #DB_PATH = 'sqlite:///' + os.path.join(RESULTS_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"Norwegian flag morphogenesis with XY inputs"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(5):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
