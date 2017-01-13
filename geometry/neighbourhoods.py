from operator import itemgetter
from typing import Sequence, Tuple, Union

COORD_1D_T = Tuple[int]
COORD_2D_T = Tuple[int, int]
COORD_T = Union[COORD_1D_T, COORD_2D_T]


def manhattan_distance(a: COORD_2D_T, b: COORD_2D_T) -> int:
    xa, ya = a
    xb, yb = b

    return abs(xa - xb) + abs(ya - yb)


def chebyshev_distance(a: COORD_2D_T, b: COORD_2D_T) -> int:
    xa, ya = a
    xb, yb = b

    return max(abs(xa - xb), abs(ya - yb))


def radius_1d(neighbourhood: Sequence[COORD_1D_T]) -> int:
    return max(
        abs(max(neighbourhood, key=itemgetter(0))[0]),
        abs(min(neighbourhood, key=itemgetter(0))[0]),
    )


def radius_2d(neighbourhood: Sequence[COORD_2D_T]) -> int:
    return max(
        abs(max(neighbourhood, key=itemgetter(0))[0]),
        abs(min(neighbourhood, key=itemgetter(0))[0]),
        abs(max(neighbourhood, key=itemgetter(1))[1]),
        abs(min(neighbourhood, key=itemgetter(1))[1]),
    )


LCR = ((-1,), (0,), (1,))
LLLCRRR = ((-3,), (-2,), (-1,), (0,), (1,), (2,), (3,))

VON_NEUMANN = tuple(
    (x, y)
    for y in range(-1, 2)
    for x in range(-1, 2)
    if manhattan_distance((x, y), (0, 0)) <= 1
)

EXTENDED_VON_NEUMANN = tuple(
    (x, y)
    for y in range(-2, 3)
    for x in range(-2, 3)
    if manhattan_distance((x, y), (0, 0)) <= 2
)

MOORE = tuple(
    (x, y)
    for y in range(-1, 2)
    for x in range(-1, 2)
    if chebyshev_distance((x, y), (0, 0)) <= 1
)
