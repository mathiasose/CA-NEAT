from neat.nn import FeedForwardNetwork

from ca_neat.config import CAConfig


def replication_fitness_f(phenotype: FeedForwardNetwork, ca_config: CAConfig) -> float:
    from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
    from ca_neat.patterns.replicate_pattern import find_pattern_partial_matches
    from ca_neat.geometry.cell_grid import CellGrid2D
    from statistics import mean
    from ca_neat.utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from typing import Sequence, Iterator
    from ca_neat.geometry.cell_grid import CELL_STATE_T

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

    def ca_develop(network: FeedForwardNetwork) -> Iterator[CellGrid2D]:
        def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
            if all((x == initial_grid.dead_cell) for x in inputs_discrete_values):
                return initial_grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield initial_grid

        for grid in iterate_ca_n_times_or_until_cycle_found(initial_grid, transition_f, iterations):
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


def morphogenesis_fitness_f(phenotype: FeedForwardNetwork, ca_config: CAConfig) -> float:
    from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
    from ca_neat.patterns.replicate_pattern import count_correct_cells
    from ca_neat.geometry.cell_grid import ToroidalCellGrid2D
    from ca_neat.utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from math import exp
    from typing import Sequence, Iterator
    from ca_neat.geometry.cell_grid import CELL_STATE_T

    neighbourhood = ca_config.neighbourhood
    alphabet = ca_config.alphabet
    target_pattern = ca_config.etc['target_pattern']
    seed = ca_config.etc['seed']
    iterations = ca_config.iterations
    state_normalization_rules = create_state_normalization_rules(states=alphabet)

    pattern_w = len(target_pattern[0])
    pattern_h = len(target_pattern)
    pattern_area = pattern_h * pattern_w

    initial_grid = ToroidalCellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        x_range=(0, pattern_w),
        y_range=(0, pattern_h),
    )

    initial_grid.add_pattern_at_coord(seed, (0, 0))

    def ca_develop(network: FeedForwardNetwork) -> Iterator[ToroidalCellGrid2D]:
        def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
            if all((x == initial_grid.dead_cell) for x in inputs_discrete_values):
                return initial_grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield initial_grid

        for grid in iterate_ca_n_times_or_until_cycle_found(initial_grid, transition_f, iterations):
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
