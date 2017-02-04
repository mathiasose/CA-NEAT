from typing import Tuple

from ca_neat.geometry.cell_grid import CELL_STATE_T

PATTERN_T = Tuple[Tuple[CELL_STATE_T, ...], ...]

QUIESCENT = ' '
ALPHABET_2 = (QUIESCENT, '■',)
ALPHABET_3 = ALPHABET_2 + ('□',)
ALPHABET_4 = ALPHABET_3 + ('▨',)

SEED_5X5 = (
    (' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', '■', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ',),
)

SEED_6X6 = (
    (' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', '■', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ',),
)

SEED_7X7 = (
    (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', '■', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
    (' ', ' ', ' ', ' ', ' ', ' ', ' ',),
)

MOSAIC = (
    (' ', '■', ' ', '■', ' ',),
    ('■', ' ', '■', ' ', '■',),
    (' ', '■', ' ', '■', ' ',),
    ('■', ' ', '■', ' ', '■',),
    (' ', '■', ' ', '■', ' ',),
)

BORDER = (
    (' ', '■', '■', '■', '■', ' ',),
    (' ', '■', '■', '■', '■', ' ',),
    (' ', '■', '■', '■', '■', ' ',),
    (' ', '■', '■', '■', '■', ' ',),
    (' ', '■', '■', '■', '■', ' ',),
    (' ', '■', '■', '■', '■', ' ',),
)

TRICOLOR = (
    ('■', '■', '□', '□', '▨', '▨',),
    ('■', '■', '□', '□', '▨', '▨',),
    ('■', '■', '□', '□', '▨', '▨',),
    ('■', '■', '□', '□', '▨', '▨',),
    ('■', '■', '□', '□', '▨', '▨',),
    ('■', '■', '□', '□', '▨', '▨',),
)

SWISS = (
    ('■', '■', '■', '■', '■',),
    ('■', '■', ' ', '■', '■',),
    ('■', ' ', ' ', ' ', '■',),
    ('■', '■', ' ', '■', '■',),
    ('■', '■', '■', '■', '■',),
)

NORWEGIAN = (
    ('■', '□', '▨', '□', '■', '■', '■',),
    ('■', '□', '▨', '□', '■', '■', '■',),
    ('□', '□', '▨', '□', '□', '□', '□',),
    ('▨', '▨', '▨', '▨', '▨', '▨', '▨',),
    ('□', '□', '▨', '□', '□', '□', '□',),
    ('■', '□', '▨', '□', '■', '■', '■',),
    ('■', '□', '▨', '□', '■', '■', '■',),
)


def pad_pattern(pattern: PATTERN_T, dead_cell: CELL_STATE_T, n: int = 1) -> PATTERN_T:
    pattern_w = len(pattern[0])

    new_w = pattern_w + 2 * n

    padded_pattern = [(dead_cell,) * new_w] * n
    padded_pattern.extend((dead_cell,) * n + row + (dead_cell,) * n for row in pattern)
    padded_pattern.extend([(dead_cell,) * new_w] * n)

    return tuple(tuple(row) for row in padded_pattern)


if __name__ == '__main__':
    from visualization.colors import colormap, norm
    from matplotlib import pyplot as plt

    import seaborn

    seaborn.set_style('white')

    patterns = MOSAIC, BORDER, TRICOLOR, SWISS, NORWEGIAN

    colors = {
        ' ': 0,
        '■': 1,
        '□': 2,
        '▨': 3,
    }

    for pattern in patterns:
        fig = plt.figure()

        plt.imshow(
            [[colors.get(x) for x in row] for row in pattern],
            interpolation='none',
            cmap=colormap,
            norm=norm,
        )

    plt.show()
