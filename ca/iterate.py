from ca.rule_tables import table_from_string
from config import CAConfig
from geometry.cell_grid import CellGrid2D
from utils import random_bitstring


def iterate_ca(grid, transition_f):
    new = grid.empty_copy()

    for coord in grid.iterate_coords():
        neighbourhood_values = grid.get_neighbourhood_values(coord)
        new.set(coord, transition_f(neighbourhood_values))

    return new


def n_iterations(initial_grid, transition_f, n):
    grid = initial_grid

    for _ in range(n):
        grid = iterate_ca(grid, transition_f=transition_f)

    return grid


def iterate_until_stable(initial_grid, transition_f, max_n):
    grid = initial_grid

    for _ in range(max_n):
        new = iterate_ca(grid, transition_f=transition_f)

        if new == grid:
            return new

        grid = new

    return grid


def ca_develop(phenotype, ca_config: CAConfig):
    from geometry.cell_grid import FiniteCellGrid2D
    from utils import make_step_f

    neighbourhood = ca_config.neighbourhood
    alphabet = ca_config.alphabet
    r = ca_config.r
    iterations = ca_config.iterations

    initial = FiniteCellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        x_range=(-r, r),
        y_range=(-r, r),
        values={(0, 0): 1},
    )

    step = make_step_f(0.5)

    def transition_f(args):
        t = tuple(int(x) for x in args)
        return step(phenotype.serial_activate(t)[0])

    grid = n_iterations(initial_grid=initial, transition_f=transition_f, n=iterations)

    return grid


if __name__ == '__main__':
    grid = CellGrid2D(cell_states='01')
    grid.set((0, 0), '1')
    transition_f = lambda k: table_from_string(random_bitstring(2 ** 5), '01')[tuple(k)]

    new = n_iterations(grid, transition_f, 5)
    print(new)
