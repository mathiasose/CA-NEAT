import operator
import os
from random import choice, getrandbits, randrange
from string import ascii_lowercase, ascii_uppercase
from typing import Any, Sequence

PROJECT_ROOT = os.path.abspath(
    os.path.join(__file__, '..')
)

bit_flip = lambda b: '0' if b == '1' else '1'


def format_bits(bits_str, justify=8, **kwargs):
    return bits_str.lstrip('0b').rjust(justify, '0')


def random_bitstring(bits=8, **kwargs):
    return format_bits(bin(getrandbits(bits)), justify=bits)


def random_string(alphabet, length, **kwargs):
    return ''.join(choice(alphabet) for _ in range(length))


def invert_value(value, alphabet):
    a, b = alphabet
    return a if value == b else b


def invert_pattern(pattern, alphabet):
    return list(map(lambda x: invert_value(x, alphabet), pattern))


def splice(a, b, **kwargs):
    i = randrange(len(a))
    return a[:i] + b[i:]


def mutate_bit(genotype, **kwargs):
    i = randrange(len(genotype) - 1)
    return genotype[:i] + bit_flip(genotype[i]) + genotype[i + 1:]


def mutate_char(genotype, alphabet, **kwargs):
    i = randrange(len(genotype) - 1)
    return genotype[:i] + choice(alphabet) + genotype[i + 1:]


def char_set(size, **kwargs):
    return (ascii_uppercase + ascii_lowercase)[:size]


def is_whole_number(n):
    return n == int(n)


def tuple_add(a, b):
    return tuple(map(operator.add, a, b))


def pluck(collection, attr):
    return (getattr(item, attr) for item in collection)


def is_even(n: int) -> int:
    return (n % 2) == 0


def is_odd(n: int) -> int:
    return not is_even(n)


def create_state_normalization_rules(states, range=(-1, 1)) -> dict:
    lo, hi = range
    n_states = len(states)
    step = (hi - lo) / n_states

    return {s: lo + i * step for i, s in enumerate(states)}


def is_all_same(Xs: Sequence[Any]) -> bool:
    return all(x == Xs[0] for x in Xs)


if __name__ == '__main__':
    d = create_state_normalization_rules('abcd')
    print(d.items())
