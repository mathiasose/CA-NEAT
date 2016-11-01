from config import CAConfig
from geometry.cell_grid import CellGrid


def iterate_ca(grid, transition_f):
    new = grid.empty_copy()

    for coord in grid.iterate_coords():
        neighbourhood_values = grid.get_neighbourhood_values(coord)
        new.set(coord, transition_f(neighbourhood_values))

    return new


def n_iterations(initial_grid, transition_f, n):
    grid = initial_grid

    for _ in range(n):
        new = iterate_ca(grid, transition_f=transition_f)
        yield grid

        if new == grid:
            break
        else:
            grid = new


def iterate_until_stable(initial_grid, transition_f, max_n):
    grid = initial_grid

    for _ in range(max_n):
        new = iterate_ca(grid, transition_f=transition_f)

        if new == grid:
            return new

        grid = new

    return grid


def ca_develop(phenotype, ca_config: CAConfig, initial_grid: CellGrid):
    from utils import create_state_normalization_rules

    iterations = ca_config.iterations

    state_normalization_rules = create_state_normalization_rules(states=ca_config.alphabet)

    def transition_f(inputs):
        inputs = tuple(inputs)

        if all((x == initial_grid.dead_cell) for x in inputs):
            return initial_grid.dead_cell

        inputs_float_values = tuple(state_normalization_rules.get_key_for_value(x) for x in inputs)

        return state_normalization_rules.get(phenotype.serial_activate(inputs_float_values)[0])

    grid_iterations = [initial_grid] + list(
        n_iterations(initial_grid=initial_grid, transition_f=transition_f, n=iterations))

    return grid_iterations
