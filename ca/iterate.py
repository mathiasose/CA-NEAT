from typing import Iterator

from geometry.cell_grid import CellGrid


def iterate_ca_once(grid: CellGrid, transition_f) -> CellGrid:
    new = grid.empty_copy()

    for coord in grid.iterate_coords():
        neighbourhood_values = grid.get_neighbourhood_values(coord)
        new.set(coord, transition_f(neighbourhood_values))

    return new


def iterate_ca_n_times(initial_grid: CellGrid, transition_f, n: int) -> Iterator[CellGrid]:
    grid = initial_grid

    for _ in range(n):
        new = iterate_ca_once(grid, transition_f=transition_f)
        yield new
        grid = new


def iterate_ca_n_times_or_until_cycle_found(initial_grid: CellGrid, transition_f, n: int) -> Iterator[CellGrid]:
    seen = {initial_grid}

    for new in iterate_ca_n_times(initial_grid, transition_f, n):
        yield new

        if new in seen:
            # When a cycle is found, the function will terminate, but not before yielding the state that was repeated.
            # The function that is calling this function can enumerate and compare the outputs to determine cycle length.
            return
        else:
            seen.add(new)
