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
            # found a cycle
            return
        else:
            yield new
            seen.add(new)
            grid = new
