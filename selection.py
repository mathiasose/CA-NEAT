from operator import attrgetter
from random import random


def roulette(population, scaling_func, **kwargs):
    """
    generates pairs with the roulette method.
    requires a scaling_func to scale the "slices"
    """

    def generate_roulette():
        """
        returns a function that when called will select an indidividual from the population
        based on the normalized fitness value as decided by fitness_func
        """

        scaled_fitnesses = list((x.individual_number, scaling_func(x.fitness)) for x in population)

        scaled_fitnesses.sort(key=lambda x: x[1])

        roulette = dict()

        current_sum = 0.0
        for individual_number, p in scaled_fitnesses:
            next_sum = current_sum + p
            roulette[(current_sum, next_sum)] = individual_number
            current_sum = next_sum

        sorted_by_lower_bound = sorted(roulette.keys(), key=lambda x: x[0])

        def get_one():
            r = random()

            for (lo, hi) in sorted_by_lower_bound:
                if lo <= r < hi:
                    return roulette[(lo, hi)]

        return get_one

    spin = generate_roulette()

    individuals_by_number = {x.individual_number: x for x in population}

    while True:
        a = b = spin()

        while a == b:
            b = spin()

        if random() < 0.5:
            a, b = b, a

        yield (individuals_by_number.get(a), individuals_by_number.get(b))


def fitness_proportionate(population, **kwargs):
    total_fitness = sum(x.fitness for x in population)
    scaling_func = lambda fitness: fitness / total_fitness

    return roulette(population=population, scaling_func=scaling_func, **kwargs)


def sigma_scaled(sigma, average_fitness, population, **kwargs):
    expected_value_func = lambda x: 1 if sigma == 0 else 1 + ((x.fitness - average_fitness) / (2 * sigma))
    sigma_sum = sum(expected_value_func(x) for x in population)
    scaling_func = lambda x: expected_value_func(x) / sigma_sum

    return roulette(population=population, scaling_func=scaling_func, **kwargs)


def ranked(population, **kwargs):
    min_f = min(population, key=attrgetter('fitness'))
    max_f = max(population, key=attrgetter('fitness'))
    sorted_population = sorted(population, key=attrgetter('fitness'), reverse=True)
    scaling_func = lambda x: min_f + (max_f - min_f) * sorted_population.index(x) / (len(population) - 1)

    return roulette(population=population, scaling_func=scaling_func, **kwargs)


def tournament(population, group_size, epsilon, **kwargs):
    def get_one(group):
        r = random.random()

        if r < epsilon:
            return random.choice(group)

        return max(group, key=attrgetter('fitness'))

    while True:
        pool = list(population)

        group_a = random.sample(pool, group_size)
        a = get_one(group_a)
        pool.remove(a)

        group_b = random.sample(pool, group_size)
        b = get_one(group_b)

        yield (a, b)


def eugenics(population, **kwargs):
    """
    Variation on deterministic uniform where the two best are paired,
    then the next two,
    and so on.
    """
    s = sorted(population, key=attrgetter('fitness'), reverse=True)

    pairs = list()
    n_pairs = len(population) / 2
    while len(pairs) < n_pairs:
        a = s.pop(0)
        b = s.pop(0)

        yield (a, b)
