from ca_neat.config import CAConfig
from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.problems.synchronization import CA_CONFIG, NEAT_CONFIG
from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
from ca_neat.utils import create_state_normalization_rules
from operator import itemgetter
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype
from typing import Iterator
from ca_neat.geometry.cell_grid import FiniteCellGrid1D
from math import exp
from ca_neat.geometry.cell_grid import CELL_STATE_T
from typing import Sequence
from ca_neat.utils import is_all_same

N = 10


def fitness_f(phenotype: FeedForwardNetwork, ca_config: CAConfig):
    alphabet = ca_config.alphabet

    iterations = ca_config.iterations
    state_normalization_rules = create_state_normalization_rules(states=alphabet)

    def ca_develop(grid, network: FeedForwardNetwork) -> Iterator[FiniteCellGrid1D]:
        def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
            if is_all_same(inputs_discrete_values):
                return inputs_discrete_values[0]

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield grid

        for grid_it in iterate_ca_n_times_or_until_cycle_found(grid, transition_f, iterations):
            yield grid_it

    pattern = [2, *([1] * 9)]
    firing_state = ca_config.alphabet[-1]

    init_grid = FiniteCellGrid1D(
        cell_states=alphabet,
        x_range=(0, len(pattern)),
        values={(i,): x for i, x in enumerate(pattern)},
        neighbourhood=ca_config.neighbourhood,
    )

    for it in ca_develop(init_grid, phenotype):
        whole = it.get_whole()
        print(whole)

        if firing_state in whole:
            break


if __name__ == '__main__':
    DB_PATH = 'postgresql+psycopg2:///synchronization_2017-04-02T22:23:29.295715'
    db = get_db(DB_PATH)

    session = db.Session()

    assert session.query(Individual).count()

    individual = session.query(Individual) \
        .order_by(-Individual.fitness) \
        .first()

    assert individual

    print(individual.fitness)

    gt = deserialize_gt(individual.genotype, NEAT_CONFIG)
    pt = create_feed_forward_phenotype(gt)
    fitness_f(pt, CA_CONFIG)
