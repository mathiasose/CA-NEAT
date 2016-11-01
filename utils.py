from string import ascii_lowercase, ascii_uppercase

import operator
from random import choice, getrandbits, randrange

bit_flip = lambda b: '0' if b == '1' else '1'


def format_bits(bits_str, justify=8, **kwargs):
    return bits_str.lstrip('0b').rjust(justify, '0')


def random_bitstring(bits=8, **kwargs):
    return format_bits(bin(getrandbits(bits)), justify=bits)


def random_string(alphabet, length, **kwargs):
    return ''.join(choice(alphabet) for _ in range(length))


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


class RangeDict(dict):
    def get(self, key: float, default=None):
        sorted_keys = sorted(self.keys(), reverse=True)
        for k in sorted_keys:
            if key > k:
                return super().get(k)
        else:
            return super().get(sorted_keys[-1])

    def get_key_for_value(self, value) -> float:
        for k, v in self.items():
            if v == value:
                return k
        else:
            return None

def create_state_normalization_rules(states, range=(-1, 1)):
    lo, hi = range
    n_states = len(states)
    step = (hi - lo) / n_states

    return RangeDict({lo + i * step: s for i, s in enumerate(states)})
