from operator import itemgetter


def manhattan_distance(a, b):
    xa, ya = a
    xb, yb = b

    return abs(xa - xb) + abs(ya - yb)


def chebyshev_distance(a, b):
    xa, ya = a
    xb, yb = b

    return max(abs(xa - xb), abs(ya - yb))


def radius_1d(neighbourhood):
    return max(
        abs(max(neighbourhood, key=itemgetter(0))[0]),
        abs(min(neighbourhood, key=itemgetter(0))[0]),
    )


def radius_2d(neighbourhood):
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
