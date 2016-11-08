from statistics import mean
from typing import List
from uuid import uuid4

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


def find_pattern_partial_matches(grid: CellGrid2D, pattern) -> List[float]:
    live_cells = set(coord for coord, _ in grid.get_live_cells())

    pattern_h, pattern_w = len(pattern), len(pattern[0])
    pattern_area = pattern_h * pattern_w

    results_by_uuid = {}
    uuid_matrix = {}

    for (x, y) in live_cells:
        x_range = (x - 1, x - 1 + pattern_w)
        y_range = (y - 1, y - 1 + pattern_h)
        rectangle = grid.get_rectangle(
            x_range=x_range,
            y_range=y_range
        )

        c = 0
        for row_a, row_b in zip(pattern, rectangle):
            for a, b in zip(row_a, row_b):
                if a == b:
                    c += 1

        match_fraction = c / pattern_area

        new_uuid = uuid4().hex
        results_by_uuid[new_uuid] = match_fraction

        conflicts = set()
        for yy in range(*y_range):
            for xx in range(*x_range):
                uuid = uuid_matrix.get((xx, yy), )
                if uuid:
                    conflicts.add(uuid)

        do_replacement = True
        for uuid in conflicts:
            value = results_by_uuid[uuid]

            if value > match_fraction:
                do_replacement = False
                break

        if do_replacement:
            for key, value in tuple(uuid_matrix.items()):
                if value in conflicts:
                    del uuid_matrix[key]

            for yy in range(*y_range):
                for xx in range(*x_range):
                    uuid_matrix[(xx, yy)] = new_uuid

    return list(map(results_by_uuid.get, set(uuid_matrix.values())))


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
    grid.add_pattern_at_coord(pattern, (10, 0))
    grid.add_pattern_at_coord(wrong_pattern, (0, 10))
    grid.add_pattern_at_coord(pattern, (10, 10))

    print(grid)

    l = find_pattern_partial_matches(grid, pattern)
    print(l)
    print(mean(l))
    print(count_pattern(grid, pattern))
