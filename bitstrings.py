from random import getrandbits, randrange


def random_individual(bits=8):
    return bin(getrandbits(bits)).lstrip('0b')


def splice(a, b):
    i = randrange(len(a))
    return a[:i] + b[i:]


def flip(bitchar):
    return '1' if bitchar == '0' else '0'


def mutate(bitstr):
    i = randrange(len(bitstr) - 1)
    return bitstr[:i] + flip(bitstr[i]) + bitstr[i + 1:]
