from typing import Iterator, List

from neat.nn import FeedForwardNetwork

from config import CAConfig
from geometry.cell_grid import CellGrid


def iterate_ca(grid: CellGrid, transition_f) -> CellGrid:
    new = grid.empty_copy()

    for coord in grid.iterate_coords():
        neighbourhood_values = grid.get_neighbourhood_values(coord)
        new.set(coord, transition_f(neighbourhood_values))

    return new


def n_iterations(initial_grid: CellGrid, transition_f, n: int) -> Iterator[CellGrid]:
    grid = initial_grid

    seen = {initial_grid}

    for _ in range(n):
        new = iterate_ca(grid, transition_f=transition_f)

        if new in seen:
            return
        else:
            yield new
            seen.add(new)
            grid = new


def ca_develop(network: FeedForwardNetwork, ca_config: CAConfig, initial_grid: CellGrid) -> Iterator[CellGrid]:
    from utils import create_state_normalization_rules

    state_normalization_rules = create_state_normalization_rules(states=ca_config.alphabet)

    def transition_f(inputs):
        inputs = tuple(inputs)

        if all((x == initial_grid.dead_cell) for x in inputs):
            return initial_grid.dead_cell

        inputs_float_values = tuple(state_normalization_rules.get_key_for_value(x) for x in inputs)

        return state_normalization_rules.get(network.serial_activate(inputs_float_values)[0])

    iterations = ca_config.iterations

    yield initial_grid

    for grid in n_iterations(initial_grid, transition_f, iterations):
        yield grid
