from utils import is_even


def evaluate(grid, grid_r, grid_cell=1, checker_colors=('0', '1')):
    correct_count = 0
    for y in range(-grid_r, grid_r):
        for x in range(-grid_r, grid_r):
            v = grid.get((x, y))
            if v not in checker_colors:
                continue

            equal_even = is_even(x // grid_cell) == is_even(y // grid_cell)
            if int(equal_even) == int(v):
                correct_count += 1

    return correct_count / grid.area
