from random import getrandbits, randrange
from string import ascii_lowercase, ascii_uppercase

bit_flip = lambda b: '0' if b == '1' else '1'


def random_bitstring(bits=8):
    return bin(getrandbits(bits)).lstrip('0b').rjust(bits, '0')


def splice(a, b):
    i = randrange(len(a))
    return a[:i] + b[i:]


def mutate(genotype):
    i = randrange(len(genotype) - 1)
    return genotype[:i] + bit_flip(genotype[i]) + genotype[i + 1:]


def char_set(size):
    return (ascii_uppercase + ascii_lowercase)[:size]
