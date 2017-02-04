from statistics import mean
from typing import Iterator

from ca_neat.geometry.cell_grid import CellGrid2D, FiniteCellGrid2D
from ca_neat.patterns.patterns import PATTERN_T


def count_pattern(grid: CellGrid2D, pattern: PATTERN_T) -> int:
    (x_min, y_min), (x_max, y_max) = grid.get_extreme_coords(pad=1)

    pattern_h, pattern_w = len(pattern), len(pattern[0])

    count = 0
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            rectangle = grid.get_rectangle(
                x_range=(x, x + pattern_w),
                y_range=(y, y + pattern_h)
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


def find_pattern_partial_matches(grid: CellGrid2D, pattern: PATTERN_T) -> Iterator[float]:
    (x_min, y_min), (x_max, y_max) = grid.get_extreme_coords(pad=1)

    pattern_h, pattern_w = len(pattern), len(pattern[0])
    pattern_area = pattern_h * pattern_w

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            rectangle = grid.get_rectangle(
                x_range=(x, x + pattern_w),
                y_range=(y, y + pattern_h),
            )

            if all(all(x == grid.dead_cell for x in row) for row in rectangle):
                yield 0.0

            correct_count = count_correct_cells(test_pattern=rectangle, target_pattern=pattern)

            yield (correct_count / pattern_area)


if __name__ == '__main__':
    cell_states = (' ', '■', '□', '▨')
    grid = FiniteCellGrid2D(cell_states=cell_states, x_range=(-10, 20), y_range=(-10, 20))
    pattern = (
        (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
        (' ', ' ', '■', ' ', '■', ' ', ' ',),
        (' ', '■', ' ', '■', ' ', '■', ' ',),
        (' ', ' ', '■', ' ', '■', ' ', ' ',),
        (' ', '■', ' ', '■', ' ', '■', ' ',),
        (' ', ' ', '■', ' ', '■', ' ', ' ',),
        (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    )
    wrong_pattern = (
        (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
        (' ', ' ', '■', ' ', '■', ' ', ' ',),
        (' ', '■', ' ', '■', ' ', '■', ' ',),
        (' ', ' ', '■', '■', '■', ' ', ' ',),
        (' ', '■', ' ', '■', ' ', '■', ' ',),
        (' ', ' ', '■', ' ', '■', ' ', ' ',),
        (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    )

    grid.add_pattern_at_coord(pattern, (0, 0))
    grid.add_pattern_at_coord(pattern, (9, 0))
    grid.add_pattern_at_coord(wrong_pattern, (0, 9))
    grid.add_pattern_at_coord(pattern, (-9, -9))

    print(grid)

    l = tuple(find_pattern_partial_matches(grid, pattern))
    print(l)
    l = tuple(sorted(l, reverse=True)[:4])
    print(l)
    print(mean(l))
    print(count_pattern(grid, pattern))
