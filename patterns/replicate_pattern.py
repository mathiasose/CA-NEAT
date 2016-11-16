from statistics import mean
from typing import Iterator

from geometry.cell_grid import CellGrid2D, FiniteCellGrid2D


def count_pattern(grid: CellGrid2D, pattern) -> int:
    live_cells = grid.get_live_cells()

    pattern_h, pattern_w = len(pattern), len(pattern[0])

    count = 0
    for ((x, y), _) in live_cells:
        rectangle = grid.get_rectangle(
            x_range=(x - 1, x - 1 + pattern_w),
            y_range=(y - 1, y - 1 + pattern_h)
        )

        if rectangle == pattern:
            count += 1

    return count


def count_correct_cells(test_pattern, target_pattern) -> int:
    correct_count = 0
    for row_a, row_b in zip(target_pattern, test_pattern):
        for a, b in zip(row_a, row_b):
            if a == b:
                correct_count += 1

    return correct_count


def find_pattern_partial_matches(grid: CellGrid2D, pattern) -> Iterator[float]:
    live_cells = set(coord for coord, _ in grid.get_live_cells())

    pattern_h, pattern_w = len(pattern), len(pattern[0])
    pattern_area = pattern_h * pattern_w

    for (x, y) in live_cells:
        x_range = (x - 1, x - 1 + pattern_w)
        y_range = (y - 1, y - 1 + pattern_h)
        rectangle = grid.get_rectangle(
            x_range=x_range,
            y_range=y_range,
        )

        correct_count = count_correct_cells(test_pattern=rectangle, target_pattern=pattern)

        yield (correct_count / pattern_area)


if __name__ == '__main__':
    cell_states = (' ', '■', '□', '▨')
    grid = FiniteCellGrid2D(cell_states=cell_states, x_range=(-10, 20), y_range=(-10, 20))
    pattern = (
        (' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',),
        (' ', '■', '□', '▨', '□', '■', '■', '■', ' ',),
        (' ', '■', '□', '▨', '□', '■', '■', '■', ' ',),
        (' ', '□', '□', '▨', '□', '□', '□', '□', ' ',),
        (' ', '▨', '▨', '▨', '▨', '▨', '▨', '▨', ' ',),
        (' ', '□', '□', '▨', '□', '□', '□', '□', ' ',),
        (' ', '■', '□', '▨', '□', '■', '■', '■', ' ',),
        (' ', '■', '□', '▨', '□', '■', '■', '■', ' ',),
        (' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    )
    wrong_pattern = (
        (' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',),
        (' ', '■', '□', '▨', '□', '■', '■', '■', ' ',),
        (' ', '■', '□', '▨', '□', '■', '■', '■', ' ',),
        (' ', '▨', '▨', '▨', '▨', '▨', '▨', '▨', ' ',),
        (' ', '□', '□', '▨', '□', '□', '□', '□', ' ',),
        (' ', '▨', '▨', '▨', '▨', '▨', '▨', '▨', ' ',),
        (' ', '■', '□', '▨', '□', '■', '■', '■', ' ',),
        (' ', '■', '□', '▨', '□', '■', '■', '■', ' ',),
        (' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    )

    grid.add_pattern_at_coord(pattern, (0, 0))
    grid.add_pattern_at_coord(pattern, (9, 0))
    grid.add_pattern_at_coord(wrong_pattern, (0, 9))
    grid.add_pattern_at_coord(pattern, (-9, -9))

    print(grid)

    l = tuple(find_pattern_partial_matches(grid, pattern))
    print(l)
    l = sorted(l, reverse=True)[:4]
    print(l)
    print(mean(l))
    print(count_pattern(grid, pattern))
