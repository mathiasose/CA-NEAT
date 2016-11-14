import os
from datetime import datetime

from config import CAConfig, CPPNNEATConfig
from geometry.neighbourhoods import VON_NEUMANN
from run_experiment import initialize_scenario
from selection import sigma_scaled


def fitness_f(phenotype, ca_config: CAConfig):
    from ca.iterate import n_iterations
    from patterns.replicate_pattern import find_pattern_partial_matches
    from geometry.cell_grid import CellGrid2D
    from statistics import mean
    from utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from typing import Tuple, T

    neighbourhood = ca_config.neighbourhood
    alphabet = ca_config.alphabet
    pattern = ca_config.etc['pattern']
    wanted_occurrences = ca_config.etc['wanted_occurrences']
    iterations = ca_config.iterations
    state_normalization_rules = create_state_normalization_rules(states=alphabet)

    initial_grid = CellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
    )

    initial_grid.add_pattern_at_coord(pattern, (0, 0))

    def ca_develop(network: FeedForwardNetwork):
        def transition_f(inputs_discrete_values: Tuple[T]) -> T:
            if all((x == initial_grid.dead_cell) for x in inputs_discrete_values):
                return initial_grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield initial_grid

        for grid in n_iterations(initial_grid, transition_f, iterations):
            yield grid

    grid_iterations = ca_develop(phenotype)

    best = 0.0
    for i, grid in enumerate(grid_iterations):
        if i == 0:
            # the initial state should not be evaluated and contribute to the score
            continue

        partial_matches = tuple(find_pattern_partial_matches(grid, pattern))

        if not partial_matches:
            continue

        best_n_matches = sorted(partial_matches, reverse=True)[:wanted_occurrences]

        # to encourage perfect replicas we penalize imperfect replicas a little bit extra
        # so that the difference between perfect and near-perfect is greater
        penalty_factor = 0.9
        best_n_matches = [(1.0 if score >= 1.0 else score * penalty_factor) for score in best_n_matches]

        avg = mean(best_n_matches)
        best = max(best, avg)

        if best >= 1.0:
            break

    return best


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = (' ', '■',)
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 25
CA_CONFIG.etc = {
    'pattern': (
        (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
        (' ', '■', '■', '■', '■', '■', ' ',),
        (' ', '■', '■', ' ', '■', '■', ' ',),
        (' ', '■', ' ', ' ', ' ', '■', ' ',),
        (' ', '■', '■', ' ', '■', '■', ' ',),
        (' ', '■', '■', '■', '■', '■', ' ',),
        (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    ),
    'wanted_occurrences': 3,
}

NEAT_CONFIG = CPPNNEATConfig()
NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 200
NEAT_CONFIG.stagnation_limit = 15
NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0
NEAT_CONFIG.weight_stdev = 1.0
NEAT_CONFIG.compatibility_threshold = 0.85
NEAT_CONFIG.prob_add_conn = 0.458
NEAT_CONFIG.prob_add_node = 0.185
NEAT_CONFIG.prob_delete_conn = 0.246
NEAT_CONFIG.prob_delete_node = 0.0352
NEAT_CONFIG.prob_mutate_bias = 0.0509
NEAT_CONFIG.bias_mutation_power = 2.093
NEAT_CONFIG.prob_mutate_response = 0.2
NEAT_CONFIG.response_mutation_power = 0.1
NEAT_CONFIG.prob_mutate_weight = 0.460
NEAT_CONFIG.prob_replace_weight = 0.0745
NEAT_CONFIG.weight_mutation_power = 0.825
NEAT_CONFIG.prob_mutate_activation = 0.1
NEAT_CONFIG.prob_toggle_link = 0.0138
NEAT_CONFIG.max_weight = 30
NEAT_CONFIG.min_weight = -30
NEAT_CONFIG.excess_coefficient = 1.0
NEAT_CONFIG.disjoint_coefficient = 1.0
NEAT_CONFIG.weight_coefficient = 0.4
NEAT_CONFIG.elitism = 1
NEAT_CONFIG.survival_threshold = 0.5

PAIR_SELECTION_F = sigma_scaled

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'results', PROBLEM_NAME))
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    DB_PATH = 'sqlite:///' + os.path.join(RESULTS_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"Swiss flag replication"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )
    for _ in range(10):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=fitness_f,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )

        # plot_fitnesses_over_generations(get_db(DB_PATH), scenario_id=1, title=DESCRIPTION, interval=300)
