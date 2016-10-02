from itertools import product
from math import log

from utils import format_bits, is_whole_number


def table_from_bitstring(bitstring):
    exponent = int(log(len(bitstring), 2))

    assert is_whole_number(exponent)

    return {
        tuple(format_bits(bin(i), justify=exponent)): b
        for i, b in enumerate(bitstring)
        }


def enumerate_state_space(values, n_dimensions):
    return product(values, repeat=n_dimensions)


def table_from_string(string, values):
    n_dimensions = int(log(len(string), len(values)))

    assert is_whole_number(n_dimensions)

    state_space = enumerate_state_space(values, n_dimensions)
    return {
        i: v for i, v in zip(state_space, string)
        }
