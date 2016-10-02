def manhattan_distance(a, b):
    xa, ya = a
    xb, yb = b

    return abs(xa - xb) + abs(ya - yb)


def chebyshev_distance(a, b):
    xa, ya = a
    xb, yb = b

    return max(abs(xa - xb), abs(ya - yb))


LCR = ((-1,), (0,), (1,))

VON_NEUMANN = tuple(
    (x, y)
    for x in range(-1, 2)
    for y in range(-1, 2)
    if manhattan_distance((x, y), (0, 0)) <= 1
)

EXTENDED_VON_NEUMANN = tuple(
    (x, y)
    for x in range(-2, 3)
    for y in range(-2, 3)
    if manhattan_distance((x, y), (0, 0)) <= 2
)

MOORE = tuple(
    (x, y)
    for x in range(-1, 2)
    for y in range(-1, 2)
    if chebyshev_distance((x, y), (0, 0)) <= 1
)
