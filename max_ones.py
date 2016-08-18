from random import random

from bitstrings import mutate, random_individual, splice
from raffle import Raffle

GENERATIONS = 100
POPULATION_SIZE = 16
MUTATION_CHANCE = 0.1


def max_ones(bitstr):
    return sum(c == '1' for c in bitstr)


if __name__ == '__main__':
    population = tuple(random_individual(16) for _ in range(POPULATION_SIZE))

    for gen in range(GENERATIONS):
        print(gen, population)
        raffle = Raffle(population, fitness_f=max_ones)
        next_generation = []

        for _ in range(POPULATION_SIZE):
            a, b = raffle.draw_pair()
            child = splice(a, b)

            if random() < MUTATION_CHANCE:
                child = mutate(child)

            next_generation.append(child)

        population = next_generation

    print('Final population', population)
