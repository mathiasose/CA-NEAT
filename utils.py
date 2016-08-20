from random import getrandbits, randrange

bit_flip = lambda b: '0' if b == '1' else '1'


def random_bitstring(bits=8):
    return bin(getrandbits(bits)).lstrip('0b')


def splice(a, b):
    i = randrange(len(a))
    return a[:i] + b[i:]


def mutate(bitstr):
    i = randrange(len(bitstr) - 1)
    return bitstr[:i] + bit_flip(bitstr[i]) + bitstr[i + 1:]
