from functools import lru_cache
from typing import Callable, Iterator, Sequence

from ca_neat.geometry.cell_grid import CELL_STATE_T, CellGrid

TRANSITION_F_T = Callable[[Sequence[CELL_STATE_T]], CELL_STATE_T]
ITERATE_F_T = Callable[[CellGrid, TRANSITION_F_T], CellGrid]


def iterate_ca_once(grid: CellGrid, transition_f: TRANSITION_F_T) -> CellGrid:
    new = grid.empty_copy()

    for coord in grid.iterate_coords():
        inputs = grid.get_neighbourhood_values(coord)
        output = transition_f(inputs)
        new.set(coord, output)

    return new


def iterate_ca_once_with_coord_inputs(grid: CellGrid, transition_f: TRANSITION_F_T) -> CellGrid:
    new = grid.empty_copy()

    for coord in grid.iterate_coords():
        neighbourhood_values = grid.get_neighbourhood_values(coord)
        inputs = tuple(neighbourhood_values) + coord
        output = transition_f(inputs)
        new.set(coord, output)

    return new


def iterate_ca_n_times(initial_grid: CellGrid, transition_f: TRANSITION_F_T, n: int,
                       iterate_f: ITERATE_F_T = iterate_ca_once) -> Iterator[CellGrid]:
    grid = initial_grid

    memoized_transition_f = lru_cache(maxsize=None)(transition_f)
    for _ in range(n):
        new = iterate_f(grid, transition_f=memoized_transition_f)
        yield new
        grid = new


def iterate_ca_n_times_or_until_cycle_found(initial_grid: CellGrid, transition_f: TRANSITION_F_T, n: int,
                                            iterate_f: ITERATE_F_T = iterate_ca_once) -> Iterator[CellGrid]:
    seen = {initial_grid}

    for new in iterate_ca_n_times(initial_grid, transition_f, n, iterate_f):
        yield new

        if new in seen:
            # When a cycle is found, the function will terminate, but not before yielding the state that was repeated.
            # The function that is calling this function can enumerate and compare the outputs to determine cycle length.
            return
        else:
            seen.add(new)
