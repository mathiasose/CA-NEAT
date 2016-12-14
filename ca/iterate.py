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


def n_iterations(initial_grid: CellGrid, transition_f, n: int, cycle_check=True) -> Iterator[CellGrid]:
    grid = initial_grid

    if cycle_check:
        seen = {initial_grid}

    for _ in range(n):
        new = iterate_ca(grid, transition_f=transition_f)

        if cycle_check and new in seen:
            # found a cycle, yield one more and stop
            yield new
            return
        else:
            yield new
            seen.add(new)
            grid = new
