from itertools import product
from statistics import mode

from copy import copy
from operator import itemgetter
from typing import Sequence, Tuple

from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype

from ca_neat.config import CAConfig
from ca_neat.ga.population import create_initial_population
from ca_neat.geometry.cell_grid import CELL_STATE_T
from ca_neat.utils import create_state_normalization_rules, invert_value


def serialize_cppn_rule(cppn: FeedForwardNetwork, ca_config: CAConfig) \
        -> Tuple[Sequence[Sequence[CELL_STATE_T]], Sequence[CELL_STATE_T]]:
    N = len(ca_config.neighbourhood)

    inputs = list(product(ca_config.alphabet, repeat=N))

    rules = create_state_normalization_rules(states=ca_config.alphabet)

    outputs = [
        max(
            zip(ca_config.alphabet, cppn.serial_activate([rules.get(x) for x in xs])),
            key=itemgetter(1)
        )[0] for xs in inputs
        ]

    return inputs, outputs


def calculate_lambda(cppn: FeedForwardNetwork, ca_config: CAConfig) -> float:
    K = len(ca_config.alphabet)
    N = len(ca_config.neighbourhood)
    q = ca_config.alphabet[0]

    _, outputs = serialize_cppn_rule(cppn, ca_config)

    n = sum(x == q for x in outputs)

    return (K ** N - n) / (K ** N)


def calculate_sensitivity(cppn: FeedForwardNetwork, ca_config: CAConfig) -> float:
    alphabet = ca_config.alphabet
    neighbourhood = ca_config.neighbourhood

    nbhs = list(product(alphabet, repeat=len(neighbourhood)))
    rules = create_state_normalization_rules(states=alphabet)

    n = len(nbhs)
    m = len(neighbourhood)
    quiescent = alphabet[0]

    def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
        if all((x == quiescent) for x in inputs_discrete_values):
            return quiescent

        inputs_float_values = tuple(rules[x] for x in inputs_discrete_values)

        outputs = cppn.serial_activate(inputs_float_values)

        return max(zip(alphabet, outputs), key=itemgetter(1))[0]

    invert_q = lambda p, q: [(invert_value(x, alphabet) if i == q else x) for i, x in enumerate(p)]

    s = 0
    for nbh in nbhs:
        for quiescent in range(m):
            inv_q = invert_q(nbh, quiescent)
            try:
                a = transition_f(nbh)
                b = transition_f(inv_q)
            except OverflowError:
                continue

            s += (a != b)

    mu = s / (n * m)

    return mu


def calculate_dominance(cppn: FeedForwardNetwork, ca_config: CAConfig) -> float:
    alphabet = ca_config.alphabet
    neighbourhood = ca_config.neighbourhood

    nbhs = list(product(alphabet, repeat=len(neighbourhood)))
    rules = create_state_normalization_rules(states=alphabet)

    quiescent = alphabet[0]

    def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
        if all((x == quiescent) for x in inputs_discrete_values):
            return quiescent

        inputs_float_values = tuple(rules[x] for x in inputs_discrete_values)

        outputs = cppn.serial_activate(inputs_float_values)

        return max(zip(alphabet, outputs), key=itemgetter(1))[0]

    heterogenous, homogenous = 0, 0

    for nbh in nbhs:
        try:
            output = transition_f(nbh)
        except OverflowError:
            continue

        m = mode(nbh)

        if output != m:
            continue
        elif all(x == m for x in nbh):
            homogenous += 1
        else:
            heterogenous += 1

    return 3 * homogenous + heterogenous


if __name__ == '__main__':
    from ca_neat.problems.morphogenesis.generate_border import CA_CONFIG, NEAT_CONFIG

    gt = next(create_initial_population(NEAT_CONFIG))
    pt = create_feed_forward_phenotype(gt)
    print(calculate_sensitivity(pt, CA_CONFIG))
    print(calculate_dominance(pt, CA_CONFIG))
