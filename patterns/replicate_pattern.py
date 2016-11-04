from geometry.cell_grid import CellGrid2D, FiniteCellGrid2D


def count_pattern(grid: CellGrid2D, pattern):
    live_cells = grid.get_live_cells()

    pattern_h, pattern_w = len(pattern), len(pattern[0])

    count = 0
    for ((x, y), _) in live_cells:
        rectangle = grid.get_rectangle(
            x_range=(x - 1, x - 1 + pattern_w),
            y_range=(y - 1, y - 1 + pattern_h))

        if rectangle == pattern:
            count += 1

    return count


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

    print(count_pattern(grid, pattern))
