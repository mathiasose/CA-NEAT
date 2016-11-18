from typing import T, Tuple

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
    ('□', '■', '■', '■', '■', '□',),
    ('□', '■', '■', '■', '■', '□',),
    ('□', '■', '■', '■', '■', '□',),
    ('□', '■', '■', '■', '■', '□',),
    ('□', '■', '■', '■', '■', '□',),
    ('□', '■', '■', '■', '■', '□',),
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


def pad_pattern(pattern: Tuple[T], dead_cell: T, n=1) -> Tuple[T]:
    pattern_w = len(pattern[0])

    new_w = pattern_w + 2 * n

    d = dead_cell
    padded_pattern = [(d,) * new_w] * n
    for row in pattern:
        padded_pattern.append((d,) * n + row + (d,) * n)

    padded_pattern += [(d,) * new_w] * n

    return tuple(padded_pattern)
