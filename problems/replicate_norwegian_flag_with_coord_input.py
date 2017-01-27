import os
from datetime import datetime

from neat.nn import FeedForwardNetwork

from config import CAConfig, CPPNNEATConfig
from ga.selection import sigma_scaled
from geometry.neighbourhoods import VON_NEUMANN
from patterns.patterns import ALPHABET_4, NORWEGIAN, QUIESCENT, pad_pattern
from run_experiment import initialize_scenario

CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_4
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 30
CA_CONFIG.etc = {
    'pattern': pad_pattern(NORWEGIAN, QUIESCENT),
    'wanted_occurrences': 3,
    'penalty_factor': 0.9,
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 10
NEAT_CONFIG.generations = 10
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


def replication_fitness_f_with_coord_inputs(phenotype: FeedForwardNetwork, ca_config: CAConfig) -> float:
    from ca.iterate import iterate_ca_n_times_or_until_cycle_found, TRANSITION_F_T
    from patterns.replicate_pattern import find_pattern_partial_matches
    from geometry.cell_grid import CellGrid2D, CellGrid
    from statistics import mean
    from utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from typing import Sequence, Iterator
    from geometry.cell_grid import CELL_STATE_T

    neighbourhood = ca_config.neighbourhood
    alphabet = ca_config.alphabet
    pattern = ca_config.etc['pattern']
    wanted_occurrences = ca_config.etc['wanted_occurrences']
    iterations = ca_config.iterations
    state_normalization_rules = create_state_normalization_rules(states=alphabet)
    coord_normalization_rules = create_state_normalization_rules(states=range(-2 * iterations, 2 * iterations))

    initial_grid = CellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
    )

    initial_grid.add_pattern_at_coord(pattern, (0, 0))

    def iterate_ca_once(grid: CellGrid, transition_f: TRANSITION_F_T) -> CellGrid:
        new = grid.empty_copy()

        for coord in grid.iterate_coords():
            neighbourhood_values = grid.get_neighbourhood_values(coord)
            new.set(coord, transition_f(tuple(neighbourhood_values) + coord))

        return new

    def ca_develop(network: FeedForwardNetwork) -> Iterator[CellGrid2D]:
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
                iterate_f=iterate_ca_once
        ):
            yield grid

    grid_iterations = ca_develop(phenotype)

    best = 0.0
    for i, grid in enumerate(grid_iterations):
        if i == 0:
            # the initial state should not be evaluated and contribute to the score
            continue

        partial_matches = list(find_pattern_partial_matches(grid, pattern))

        if not partial_matches:
            continue

        sorted_matches = sorted(partial_matches, reverse=True)
        extension = [0.0] * wanted_occurrences
        best_n_matches = (sorted_matches + extension)[:wanted_occurrences]

        # to encourage perfect replicas we penalize imperfect replicas a little bit extra
        # so that the difference between perfect and near-perfect is greater
        penalty_factor = ca_config.etc.get('penalty_factor', 1.0)
        best_n_matches = [(1.0 if score >= 1.0 else score * penalty_factor) for score in best_n_matches]

        avg = mean(best_n_matches)
        best = max(best, avg)

        if best >= 1.0:
            break

    return best


FITNESS_F = replication_fitness_f_with_coord_inputs

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'results', PROBLEM_NAME))
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    DB_PATH = 'sqlite:///' + os.path.join(RESULTS_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"Nordic Cross flag replication with XY inputs \npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(1):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
