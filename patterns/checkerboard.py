from typing import Sequence

from geometry.cell_grid import CellGrid2D
from utils import is_even


def evaluate(grid: CellGrid2D, grid_r: int, grid_cell: int = 1, checker_colors: Sequence[str] = ('0', '1')) -> float:
    correct_count = 0
    for y in range(-grid_r, grid_r):
        for x in range(-grid_r, grid_r):
            v = grid.get((x, y))
            if v not in checker_colors:
                continue

            equal_even = is_even(x // grid_cell) == is_even(y // grid_cell)
            if int(equal_even) == int(v):
                correct_count += 1

    ratio = correct_count / grid.area

    return ratio
