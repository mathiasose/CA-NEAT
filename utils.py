import operator
from random import choice, getrandbits, randrange
from string import ascii_lowercase, ascii_uppercase

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
