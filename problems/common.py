from config import CAConfig


def replication_fitness_f(phenotype, ca_config: CAConfig):
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

        partial_matches = list(find_pattern_partial_matches(grid, pattern))

        if not partial_matches:
            continue

        while len(partial_matches) < wanted_occurrences:
            partial_matches.append(0.0)

        best_n_matches = sorted(partial_matches, reverse=True)[:wanted_occurrences]

        assert len(best_n_matches) == wanted_occurrences

        # to encourage perfect replicas we penalize imperfect replicas a little bit extra
        # so that the difference between perfect and near-perfect is greater
        penalty_factor = ca_config.etc.get('penalty_factor', 1.0)
        best_n_matches = [(1.0 if score >= 1.0 else score * penalty_factor) for score in best_n_matches]

        avg = mean(best_n_matches)
        best = max(best, avg)

        if best >= 1.0:
            break

    return best
