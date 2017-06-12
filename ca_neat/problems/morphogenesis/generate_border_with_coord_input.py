import os
from datetime import datetime

from neat.nn import FeedForwardNetwork

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import VON_NEUMANN
from ca_neat.patterns.patterns import ALPHABET_2, BORDER, SEED_6X6
from ca_neat.run_experiment import initialize_scenario

CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_2
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 30
CA_CONFIG.etc = {
    'target_pattern': BORDER,
    'seed': SEED_6X6,
}
CA_CONFIG.compute_lambda = False

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 100
NEAT_CONFIG.elitism = 1

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood) + 2
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

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
    pattern_area = pattern_h * pattern_w

    initial_grid = ToroidalCellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        x_range=(0, pattern_w),
        y_range=(0, pattern_h),
    )

    state_normalization_rules = create_state_normalization_rules(states=alphabet)
    r = radius_2d(neighbourhood)
    coord_normalization_rules = create_state_normalization_rules(states=range(0 - r, max(pattern_h, pattern_w) + r + 1))

    initial_grid.add_pattern_at_coord(seed, (0, 0))

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
        correctness_fraction = count_correct_cells(grid.get_whole(), target_pattern=target_pattern) / pattern_area

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

    DB_PATH = 'postgresql+psycopg2:///{}_{}'.format(PROBLEM_NAME, datetime.now().isoformat())

    DESCRIPTION = '"Border morphogenesis with XY inputs"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(100):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
