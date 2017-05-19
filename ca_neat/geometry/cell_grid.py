from collections import defaultdict
from operator import itemgetter
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

from numpy.lib.function_base import rot90
from numpy.lib.twodim_base import fliplr, flipud

from ca_neat.geometry.neighbourhoods import (COORD_1D_T, COORD_2D_T, COORD_T, LCR, NEIHGBOURHOOD_1D_T,
                                             NEIHGBOURHOOD_2D_T, NEIHGBOURHOOD_T, VON_NEUMANN, radius_1d, radius_2d)
from ca_neat.utils import tuple_add

CELL_STATE_T = Union[str, int]


class CellGrid(defaultdict):
    dimensionality: int = None
    neighbourhood: NEIHGBOURHOOD_T = None
    origin: COORD_T = None

    def __init__(self, cell_states: Sequence[CELL_STATE_T], values: Optional[Dict[COORD_T, CELL_STATE_T]] = None,
                 neighbourhood: NEIHGBOURHOOD_T = None) -> None:
        self.cell_states: Sequence[CELL_STATE_T] = cell_states
        self.dead_cell: CELL_STATE_T = cell_states[0]
        super().__init__()

        if values:
            for key, value in values.items():
                self.set(key, value)

        if neighbourhood:
            self.neighbourhood = neighbourhood

    def get(self, coord: COORD_T, default: Optional[CELL_STATE_T] = None) -> CELL_STATE_T:
        assert len(coord) == self.dimensionality

        if default is None:
            default = self.dead_cell

        return super().get(coord, default)

    def set(self, coord: COORD_T, value: CELL_STATE_T) -> None:
        assert len(coord) == self.dimensionality
        assert value in self.cell_states

        if (coord != self.origin) and (value == self.dead_cell):
            if coord in self.keys():
                super().__delitem__(coord)
            return

        super().__setitem__(coord, value)

    def get_neighbourhood_values(self, coord: COORD_T) -> Sequence[CELL_STATE_T]:
        return tuple(
            self.get(tuple_add(coord, direction)) for direction in self.neighbourhood
        )

    def empty_copy(self):
        new = self.__class__(cell_states=self.cell_states)
        new.neighbourhood = self.neighbourhood
        return new

    def iterate_coords(self) -> Iterator[COORD_T]:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if self.__class__ != other.__class__:
            return False

        if len(self) != len(other):
            return False

        for a, b in zip(self.iterate_coords(), other.iterate_coords()):
            if self.get(a) != other.get(b):
                return False

        return True

    def get_extreme_coords(self, pad: int = 0) -> Sequence[COORD_T]:
        raise NotImplementedError

    def get_live_cells(self) -> Iterator[Tuple[COORD_T, CELL_STATE_T]]:
        return ((coord, cell) for (coord, cell) in self.items() if cell != self.dead_cell)

    def __hash__(self) -> int:
        return hash(tuple(self.get_live_cells()))


class CellGrid1D(CellGrid):
    dimensionality: int = 1
    neighbourhood: NEIHGBOURHOOD_1D_T = LCR
    origin: COORD_1D_T = (0,)

    def get_extreme_coords(self, pad: int = 0) -> Tuple[COORD_1D_T, COORD_1D_T]:
        x_min = min(self.keys())[0] - pad
        x_max = max(self.keys())[0] + pad

        return (x_min,), (x_max,)

    @property
    def area(self) -> int:
        (x_min,), (x_max,) = self.get_extreme_coords()

        return max(abs(x_max - x_min), 1)

    def iterate_coords(self) -> Iterator[COORD_1D_T]:
        (x_min,), (x_max,) = self.get_extreme_coords(pad=radius_1d(self.neighbourhood))

        for x in range(x_min, x_max):
            yield (x,)

    def __str__(self) -> str:
        return ''.join(
            '{}'.format(self.get(coord)) for coord in self.iterate_coords()
        )

    def get_range(self, x_range: Tuple[int, int]) -> Iterator[CELL_STATE_T]:
        l, r = x_range

        return (self.get((x,)) for x in range(l, r))


class CellGrid2D(CellGrid):
    dimensionality: int = 2
    neighbourhood: NEIHGBOURHOOD_2D_T = VON_NEUMANN
    origin: COORD_2D_T = (0, 0)

    def get_extreme_coords(self, pad: int = 0) -> Tuple[COORD_2D_T, COORD_2D_T]:
        x_min = min(self.keys(), key=itemgetter(0))[0] - pad
        x_max = max(self.keys(), key=itemgetter(0))[0] + pad
        y_min = min(self.keys(), key=itemgetter(1))[1] - pad
        y_max = max(self.keys(), key=itemgetter(1))[1] + pad

        return (x_min, y_min), (x_max, y_max)

    @property
    def area(self) -> int:
        (x_min, y_min), (x_max, y_max) = self.get_extreme_coords()

        return max(abs(x_max - x_min), 1) * max(abs(y_max - y_min), 1)

    def iterate_coords(self) -> Iterator[COORD_2D_T]:
        (x_min, y_min), (x_max, y_max) = self.get_extreme_coords(pad=radius_2d(self.neighbourhood))

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                yield (x, y)

    def __str__(self) -> str:
        (x_min, y_min), (x_max, y_max) = self.get_extreme_coords()

        s = ''
        for y in range(y_min, y_max):
            s += '{}\t|'.format(y)
            for x in range(x_min, x_max):
                v = str(self.get((x, y)))
                s += ' ' if v == self.dead_cell else v
            s += '|\n'

        return s

    def get_rectangle(self, x_range: Tuple[int, int], y_range: Tuple[int, int],
                      f: Optional[Callable[..., CELL_STATE_T]] = None) -> Sequence[Sequence[CELL_STATE_T]]:
        l, r = x_range
        t, b = y_range

        if f is None:
            return tuple(
                tuple(
                    self.get((x, y))
                    for x in range(l, r)
                ) for y in range(t, b)
            )
        else:
            return tuple(
                tuple(
                    f(
                        self.get((x, y))
                    )
                    for x in range(l, r)
                ) for y in range(t, b)
            )

    def get_enumerated_rectangle(self, x_range: Tuple[int, int], y_range: Tuple[int, int]):
        enumeration = dict((v, k) for k, v in enumerate(self.cell_states))
        return self.get_rectangle(x_range, y_range, f=enumeration.get)

    def add_pattern_at_coord(self, pattern: Sequence[Sequence[CELL_STATE_T]], coord: COORD_2D_T):
        start_x, start_y = coord

        for y, row in enumerate(pattern):
            for x, value in enumerate(row):
                self.set((start_x + x, start_y + y), value)


class FiniteCellGrid1D(CellGrid1D):
    def __init__(self, cell_states: Sequence[CELL_STATE_T], x_range: Tuple[int, int],
                 values: Optional[Dict[COORD_1D_T, CELL_STATE_T]] = None,
                 neighbourhood: NEIHGBOURHOOD_1D_T = None):
        self.x_range = x_range
        super().__init__(cell_states=cell_states, values=values, neighbourhood=neighbourhood)

    def get_extreme_coords(self, pad: int = 0):
        l, r = self.x_range
        return (l,), (r,)

    def is_coord_within_bounds(self, coord: COORD_1D_T):
        x, = coord
        l, r = self.x_range

        return l <= x < r

    def set(self, coord: COORD_1D_T, value: CELL_STATE_T):
        if not self.is_coord_within_bounds(coord):
            return

        super().set(coord, value)

    def get(self, coord: COORD_1D_T, default: Optional[CELL_STATE_T] = None):
        if not self.is_coord_within_bounds(coord):
            return default or self.dead_cell

        return super().get(coord, default=default)

    def yield_values(self):
        return (self.get((coord,)) for coord in range(*self.x_range))

    def get_whole(self, f=lambda x: x):
        return list(map(f, self.yield_values()))

    def empty_copy(self):
        new = self.__class__(cell_states=self.cell_states, x_range=self.x_range)
        new.neighbourhood = self.neighbourhood
        return new


class ToroidalCellGrid1D(FiniteCellGrid1D):
    def __init__(self, cell_states: Sequence[CELL_STATE_T], x_range: Tuple[int, int],
                 values: Optional[Dict[COORD_1D_T, CELL_STATE_T]] = None,
                 neighbourhood: NEIHGBOURHOOD_1D_T = None):
        x0, x1 = x_range

        assert x0 == 0

        super().__init__(
            cell_states=cell_states,
            x_range=x_range,
            values=values,
            neighbourhood=neighbourhood
        )

    def get_absolute_coord(self, coord: COORD_1D_T):
        x, = coord
        l, r = self.x_range

        return (x % r),

    def set(self, coord: COORD_1D_T, value: CELL_STATE_T):
        super().set(self.get_absolute_coord(coord), value)

    def get(self, coord: COORD_1D_T, default: Optional[CELL_STATE_T] = None):
        return super().get(self.get_absolute_coord(coord), default=default)


class FiniteCellGrid2D(CellGrid2D):
    def __init__(self, cell_states: Sequence[CELL_STATE_T], x_range: Tuple[int, int], y_range: Tuple[int, int],
                 values: Optional[Dict[COORD_1D_T, CELL_STATE_T]] = None, neighbourhood: NEIHGBOURHOOD_2D_T = None):
        self.x_range = x_range
        self.y_range = y_range
        super().__init__(cell_states=cell_states, values=values, neighbourhood=neighbourhood)

    def get_extreme_coords(self, pad: int = 0):
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        return (x_min - pad, y_min - pad), (x_max + pad, y_max + pad)

    def is_coord_within_bounds(self, coord: COORD_2D_T):
        x, y = coord
        l, r = self.x_range
        t, b = self.y_range

        return (l <= x < r) and (t <= y < b)

    def set(self, coord: COORD_2D_T, value: CELL_STATE_T):
        if not self.is_coord_within_bounds(coord):
            return

        super().set(coord, value)

    def get(self, coord: COORD_2D_T, default: Optional[CELL_STATE_T] = None):
        if not self.is_coord_within_bounds(coord):
            return default or self.dead_cell

        return super().get(coord, default=default)

    def get_whole(self, f: Optional[Callable[..., CELL_STATE_T]] = None):
        return self.get_rectangle(self.x_range, self.y_range, f=f)

    def get_enumerated_whole(self):
        enumeration = dict((v, k) for k, v in enumerate(self.cell_states))
        return self.get_whole(f=enumeration.get)

    def empty_copy(self):
        new = self.__class__(cell_states=self.cell_states, x_range=self.x_range, y_range=self.y_range)
        new.neighbourhood = self.neighbourhood
        return new


class ToroidalCellGrid2D(FiniteCellGrid2D):
    def __init__(self, cell_states: Sequence[CELL_STATE_T], x_range: Tuple[int, int], y_range: Tuple[int, int],
                 values: Optional[Dict[COORD_1D_T, CELL_STATE_T]] = None, neighbourhood: NEIHGBOURHOOD_2D_T = None):
        x0, x1 = x_range
        y0, y1 = y_range

        assert x0 == 0
        assert y0 == 0

        super().__init__(
            cell_states=cell_states,
            x_range=x_range,
            y_range=y_range,
            values=values,
            neighbourhood=neighbourhood
        )

    def is_coord_within_bounds(self, coord: COORD_2D_T):
        x, y = coord
        l, r = self.x_range
        t, b = self.y_range

        return (l <= x < r) and (t <= y < b)

    def get_absolute_coord(self, coord: COORD_2D_T):
        x, y = coord
        l, r = self.x_range
        t, b = self.y_range

        return x % r, y % b

    def set(self, coord: COORD_2D_T, value: CELL_STATE_T):
        super().set(self.get_absolute_coord(coord), value)

    def get(self, coord: COORD_2D_T, default: Optional[CELL_STATE_T] = None):
        return super().get(self.get_absolute_coord(coord), default=default)

    def get_rotation(self, k: int = 1):
        copy = self.empty_copy()

        copy.add_pattern_at_coord(rot90(self.get_whole(), k=k), (0, 0))

        return copy

    def get_fliplr(self):
        lr = self.empty_copy()
        lr.add_pattern_at_coord(fliplr(self.get_whole()), (0, 0))
        return lr

    def get_flipud(self):
        ud = self.empty_copy()
        ud.add_pattern_at_coord(flipud(self.get_whole()), (0, 0))
        return ud


def get_all_rotations_and_flips(r0: CellGrid2D):
    r1 = r0.get_rotation(k=1)
    r2 = r0.get_rotation(k=2)
    r3 = r0.get_rotation(k=3)

    return (
        r0, r0.get_fliplr(), r0.get_flipud(),
        r1, r1.get_fliplr(), r1.get_flipud(),
        r2, r2.get_fliplr(), r2.get_flipud(),
        r3, r3.get_fliplr(), r3.get_flipud()
    )


def get_rotational_hash(r0: CellGrid2D):
    return hash(
        sum(
            map(lambda x: x.__hash__(), get_all_rotations_and_flips(r0))
        )
    )


if __name__ == '__main__':
    grid = ToroidalCellGrid2D(cell_states='01', x_range=(0, 5), y_range=(0, 5))
    grid.add_pattern_at_coord(
        [
            ['1', '0', '1', '1', '0'],
            ['1', '0', '1', '0', '1'],
            ['1', '0', '1', '0', '1'],
            ['1', '0', '1', '0', '1'],
            ['1', '0', '1', '1', '0'],
        ],
        (0, 0)
    )
    h0 = get_rotational_hash(grid)
    h1 = get_rotational_hash(grid.get_rotation())
    assert h0 == h1

    g2 = ToroidalCellGrid2D(cell_states='01', x_range=(0, 5), y_range=(0, 5))
    g2.add_pattern_at_coord(
        [
            ['0', '1', '1', '0', '1'],
            ['1', '0', '1', '0', '1'],
            ['1', '0', '1', '0', '1'],
            ['1', '0', '1', '0', '1'],
            ['0', '1', '1', '0', '1'],
        ],
        (0, 0)
    )

    assert get_rotational_hash(g2) == h0
