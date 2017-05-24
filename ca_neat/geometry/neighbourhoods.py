from operator import itemgetter
from typing import Sequence, Tuple, Union

COORD_1D_T = Tuple[int]
COORD_2D_T = Tuple[int, int]
COORD_T = Union[COORD_1D_T, COORD_2D_T]
NEIHGBOURHOOD_1D_T = Sequence[COORD_1D_T]
NEIHGBOURHOOD_2D_T = Sequence[COORD_2D_T]
NEIHGBOURHOOD_T = Union[NEIHGBOURHOOD_1D_T, NEIHGBOURHOOD_2D_T]


def manhattan_distance(a: COORD_2D_T, b: COORD_2D_T) -> int:
    xa, ya = a
    xb, yb = b

    return abs(xa - xb) + abs(ya - yb)


def chebyshev_distance(a: COORD_2D_T, b: COORD_2D_T) -> int:
    xa, ya = a
    xb, yb = b

    return max(abs(xa - xb), abs(ya - yb))


def radius_1d(neighbourhood: NEIHGBOURHOOD_1D_T) -> int:
    return max(
        abs(max(neighbourhood, key=itemgetter(0))[0]),
        abs(min(neighbourhood, key=itemgetter(0))[0]),
    )


def radius_2d(neighbourhood: NEIHGBOURHOOD_2D_T) -> int:
    return max(
        abs(max(neighbourhood, key=itemgetter(0))[0]),
        abs(min(neighbourhood, key=itemgetter(0))[0]),
        abs(max(neighbourhood, key=itemgetter(1))[1]),
        abs(min(neighbourhood, key=itemgetter(1))[1]),
    )


def one_d_neighborhood(r):
    return tuple((x,) for x in range(-r, r + 1))


LCR = one_d_neighborhood(1)
LLLCRRR = one_d_neighborhood(3)

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


def moore(r=1):
    return tuple(
        (x, y)
        for y in range(-r, r + 1)
        for x in range(-r, r + 1)
        if chebyshev_distance((x, y), (0, 0)) <= r
    )


MOORE = moore(1)

if __name__ == '__main__':
    assert one_d_neighborhood(1) == LCR == ((-1,), (0,), (1,))
    assert one_d_neighborhood(3) == LLLCRRR == ((-3,), (-2,), (-1,), (0,), (1,), (2,), (3,))
