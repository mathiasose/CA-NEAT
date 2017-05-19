from operator import attrgetter
from random import choice, random, sample
from statistics import mean, stdev
from typing import Callable, Iterator, List, Sequence, Tuple

from neat.genome import Genome

PAIR_T = Tuple[Genome, Genome]
PAIR_SELECTION_F_T = Callable[[Sequence[Genome]], Iterator[PAIR_T]]


class TooFewIndividuals(Exception):
    pass


def roulette(population: Sequence[Genome], scaling_func: Callable[[float], float], **kwargs) -> Iterator[PAIR_T]:
    """
    generates pairs with the roulette method.
    requires a scaling_func to scale the "slices"
    """

    try:
        assert len(population) > 1
    except AssertionError:
        raise TooFewIndividuals

    def generate_roulette() -> Callable[[], Genome]:
        """
        returns a function that when called will select an indidividual from the population
        based on the normalized fitness value as decided by fitness_func
        """

        scaled_fitnesses = list((x.ID, scaling_func(x.fitness)) for x in population)

        scaled_fitnesses.sort(key=lambda x: x[1])

        roulette = dict()

        current_sum = 0.0
        for id, p in scaled_fitnesses:
            next_sum = current_sum + p
            roulette[(current_sum, next_sum)] = id
            current_sum = next_sum

        sorted_by_lower_bound = sorted(roulette.keys(), key=lambda x: x[0])

        def get_one() -> Genome:
            r = random()

            for (lo, hi) in sorted_by_lower_bound:
                if lo <= r < hi:
                    return roulette[(lo, hi)]

        return get_one

    spin = generate_roulette()

    individuals_by_id = {x.ID: x for x in population}

    while True:
        a = b = spin()

        while a == b:
            b = spin()

        yield (individuals_by_id.get(a), individuals_by_id.get(b))


def fitness_proportionate(population: List[Genome], **kwargs) -> Iterator[PAIR_T]:
    total_fitness = sum(x.fitness for x in population)
    scaling_func = lambda fitness: fitness / total_fitness

    return roulette(population=population, scaling_func=scaling_func, **kwargs)


def sigma_scaled(population: List[Genome], **kwargs) -> Iterator[PAIR_T]:
    try:
        assert len(population) > 1
    except AssertionError:
        raise TooFewIndividuals

    fitnesses = tuple(x.fitness for x in population)

    try:
        assert any(f > 0.0 for f in fitnesses)
    except AssertionError:
        return random_choice(population)

    sigma = stdev(fitnesses)

    average_fitness = mean(fitnesses)
    expected_value_func = lambda x: 1 if sigma == 0 else 1 + ((x - average_fitness) / (2 * sigma))
    sigma_sum = sum(expected_value_func(x) for x in fitnesses)
    scaling_func = lambda x: expected_value_func(x) / sigma_sum

    return roulette(population=population, scaling_func=scaling_func, **kwargs)


def ranked(population: List[Genome], **kwargs) -> Iterator[PAIR_T]:
    min_f = min(population, key=attrgetter('fitness'))
    max_f = max(population, key=attrgetter('fitness'))
    sorted_population = sorted(population, key=attrgetter('fitness'), reverse=True)
    scaling_func = lambda x: min_f + (max_f - min_f) * sorted_population.index(x) / (len(population) - 1)

    return roulette(population=population, scaling_func=scaling_func, **kwargs)


def tournament(population: List[Genome], group_size: int, epsilon: float, **kwargs) -> Iterator[PAIR_T]:
    def get_one(group: Sequence[Genome]) -> Genome:
        r = random()

        if r < epsilon:
            return choice(group)

        return max(group, key=attrgetter('fitness'))

    try:
        assert len(population) > 1
    except AssertionError:
        raise TooFewIndividuals

    while True:
        pool = list(population)  # make a shallow copy

        group_a = sample(pool, group_size)
        a = get_one(group_a)
        pool.remove(a)

        group_b = sample(pool, group_size)
        b = get_one(group_b)

        yield (a, b)


def random_choice(population: List[Genome], **kwargs) -> Iterator[PAIR_T]:
    try:
        assert len(population) > 0
    except AssertionError:
        raise TooFewIndividuals

    while True:
        a = choice(population)
        b = choice(population)

        yield (a, b)
