import os
from datetime import datetime

from config import CAConfig, CPPNNEATConfig
from geometry.neighbourhoods import VON_NEUMANN
from run_experiment import initialize_scenario
from selection import sigma_scaled
from visualization.plot_fitness import plot_fitnesses_over_generations


def fitness_f(phenotype, ca_config: CAConfig):
    from ca.iterate import ca_develop
    from patterns.replicate_pattern import find_pattern_partial_matches
    from geometry.cell_grid import CellGrid2D
    from statistics import mean

    wanted_occurences = 3

    neighbourhood = ca_config.neighbourhood
    alphabet = ca_config.alphabet

    pattern = ca_config.etc['pattern']
    initial_grid = CellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
    )

    initial_grid.add_pattern_at_coord(pattern, (0, 0))

    grid_iterations = tuple(ca_develop(phenotype, ca_config, initial_grid))

    best = 0.0
    for i, grid in enumerate(grid_iterations):
        if i == 0:
            continue

        partial_matches = find_pattern_partial_matches(grid, pattern)

        if not partial_matches:
            continue

        best = max(best, mean(partial_matches) * min(1.0, len(partial_matches) / wanted_occurences))

        if best >= 1.0:
            best = 1.0
            break

    return best


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = (' ', '■', '□', '▨')
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 25
CA_CONFIG.etc = {
    'pattern': (
        (' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',),
        (' ', '■', '■', '□', '□', '▨', '▨', ' ',),
        (' ', '■', '■', '□', '□', '▨', '▨', ' ',),
        (' ', '■', '■', '□', '□', '▨', '▨', ' ',),
        (' ', '■', '■', '□', '□', '▨', '▨', ' ',),
        (' ', '■', '■', '□', '□', '▨', '▨', ' ',),
        (' ', '■', '■', '□', '□', '▨', '▨', ' ',),
        (' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    )
}

NEAT_CONFIG = CPPNNEATConfig()
NEAT_CONFIG.pop_size = 100
NEAT_CONFIG.generations = 50
NEAT_CONFIG.stagnation_limit = 10
NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.initial_hidden_nodes = 5
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
NEAT_CONFIG.elitism = 2
NEAT_CONFIG.survival_threshold = 0.4

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'results', PROBLEM_NAME))
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    DB_PATH = 'sqlite:///' + os.path.join(RESULTS_DIR, '{}.db'.format(datetime.now()))
    print(DB_PATH)

    DESCRIPTION = '"Tricolor flag replication"\npopulation size: {pop}\ngenerations: {gens}'.format(
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
    plot_fitnesses_over_generations(DB_PATH, title=DESCRIPTION, interval=300)
