from collections import defaultdict
from operator import itemgetter

from geometry.neighbourhoods import LCR, VON_NEUMANN
from utils import tuple_add


class CellGrid(defaultdict):
    dimensionality = None
    neighbourhood = None

    def __init__(self, cell_states, values=None, neighbourhood=None):
        self.cell_states = cell_states
        self.dead_cell = cell_states[0]
        super().__init__()

        if values:
            for key, value in values:
                self.set(key, value)

        if neighbourhood:
            self.neighbourhood = neighbourhood

    def get(self, coord, default=None):
        assert len(coord) == self.dimensionality

        return super().get(coord, default or self.dead_cell)

    def set(self, coord, value):
        assert len(coord) == self.dimensionality
        assert value in self.cell_states

        if value == self.dead_cell:
            if coord in self.keys():
                super().__delitem__(coord)
            return

        super().__setitem__(coord, value)

    def get_neighbourhood_values(self, coord):
        for direction in self.neighbourhood:
            yield self.get(tuple_add(coord, direction))

    def empty_copy(self):
        new = self.__class__(cell_states=self.cell_states)
        new.neighbourhood = self.neighbourhood
        return new

    def iterate_coords(self):
        raise NotImplementedError

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        for a, b in zip(self.iterate_coords(), other.iterate_coords()):
            if self.get(a) != other.get(b):
                return False

        return True


class CellGrid1D(CellGrid):
    dimensionality = 1
    neighbourhood = LCR

    def get_extreme_coords(self):
        x_min = min(self.keys())[0] + min(self.neighbourhood, key=itemgetter(0))[0]
        x_max = max(self.keys())[0] + max(self.neighbourhood, key=itemgetter(0))[0]

        return x_min, x_max

    def iterate_coords(self):
        x_min, x_max = self.get_extreme_coords()

        for x in range(x_min, x_max + 1):
            yield (x,)

    def __str__(self):
        return ''.join(
            '{}'.format(self.get(coord))
            for coord in self.iterate_coords()
        )


class CellGrid2D(CellGrid):
    dimensionality = 2
    neighbourhood = VON_NEUMANN

    def get_extreme_coords(self):
        x_min = min(self.keys(), key=itemgetter(0))[0] + min(self.neighbourhood, key=itemgetter(0))[0]
        x_max = max(self.keys(), key=itemgetter(0))[0] + max(self.neighbourhood, key=itemgetter(0))[0]
        y_min = min(self.keys(), key=itemgetter(1))[1] + min(self.neighbourhood, key=itemgetter(1))[1]
        y_max = max(self.keys(), key=itemgetter(1))[1] + max(self.neighbourhood, key=itemgetter(1))[1]

        return (x_min, y_min), (x_max, y_max)

    def iterate_coords(self):
        (x_min, y_min), (x_max, y_max) = self.get_extreme_coords()

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                yield (x, y)

    def __str__(self):
        (x_min, y_min), (x_max, y_max) = self.get_extreme_coords()

        s = ''
        for y in range(y_min, y_max + 1):
            s += '{}\t|'.format(y)
            for x in range(x_min, x_max + 1):
                v = self.get((x, y))
                s += ' ' if v == self.dead_cell else v
            s += '|\n'

        return s


if __name__ == '__main__':
    g = CellGrid2D(cell_states='01')
    g.set((0, 0), '1')
    g.set((3, -5), '1')
    g.set((-5, 10), '1')
    g.set((-5, -5), '1')
    g.set((15, 0), '1')
    print(g)
