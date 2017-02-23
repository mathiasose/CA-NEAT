from itertools import product
from operator import itemgetter
from typing import Dict, Sequence

from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype

from ca_neat.config import CAConfig
from ca_neat.ga.population import create_initial_population
from ca_neat.geometry.cell_grid import CELL_STATE_T
from ca_neat.utils import create_state_normalization_rules


def serialize_cppn_rule(cppn: FeedForwardNetwork, ca_config: CAConfig) -> Dict[Sequence[CELL_STATE_T], CELL_STATE_T]:
    N = len(ca_config.neighbourhood)

    inputs = product(ca_config.alphabet, repeat=N)

    rules = create_state_normalization_rules(states=ca_config.alphabet)

    outputs = {
        xs: max(
            zip(ca_config.alphabet, cppn.serial_activate([rules.get(x) for x in xs])),
            key=itemgetter(1)
        )[0] for xs in inputs
        }

    return outputs


def calculate_lambda(cppn: FeedForwardNetwork, ca_config: CAConfig) -> float:
    K = len(ca_config.alphabet)
    N = len(ca_config.neighbourhood)
    q = ca_config.alphabet[0]

    outputs = serialize_cppn_rule(cppn, ca_config)

    n = sum(x == q for x in outputs.values())

    return (K ** N - n) / (K ** N)


if __name__ == '__main__':
    from ca_neat.problems.morphogenesis.generate_border import CA_CONFIG, NEAT_CONFIG

    gt = next(create_initial_population(NEAT_CONFIG))
    λ = calculate_lambda(create_feed_forward_phenotype(gt), CA_CONFIG)
    print(λ)
