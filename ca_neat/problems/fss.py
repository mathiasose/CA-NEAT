from datetime import datetime

import os
from neat.nn import FeedForwardNetwork

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import LLLCRRR, LCR
from ca_neat.run_experiment import initialize_scenario

N = 10


def fitness_f(phenotype: FeedForwardNetwork, ca_config: CAConfig) -> float:
    from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
    from ca_neat.utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from typing import Iterator
    from ca_neat.geometry.cell_grid import FiniteCellGrid1D
    from math import exp
    from ca_neat.geometry.cell_grid import CELL_STATE_T
    from typing import Sequence
    from ca_neat.utils import is_all_same

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

        # yield grid

        for grid_it in iterate_ca_n_times_or_until_cycle_found(grid, transition_f, iterations):
            yield grid_it

    N = 10
    pattern = [2, *([1] * (N - 1))]
    firing_state = alphabet[-1]

    init_grid = FiniteCellGrid1D(
        cell_states=alphabet,
        x_range=(0, len(pattern)),
        values={(i,): x for i, x in enumerate(pattern)},
        neighbourhood=ca_config.neighbourhood,
    )

    k = 1
    redistribute = lambda x: x * exp(k * x) / exp(k)

    for i, it in enumerate(ca_develop(init_grid, phenotype)):
        s = sum(x == firing_state for x in it.get_whole())

        if s > 0:
            target_t = (2 * N - 2)
            #t_score = 1.0 - min(1.0, (abs(target_t - i) / target_t))
            if i >= target_t:
                t_score = 1.0
            else:
                t_score = (target_t - i) / target_t

            fire_score = (s / init_grid.area)
            #return 0.75 * fire_score + 0.25 * redistribute(t_score)
            return fire_score * redistribute(t_score)
    else:
        return 0.0


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = tuple(range(15))
CA_CONFIG.neighbourhood = LCR
CA_CONFIG.iterations = 10 * N
CA_CONFIG.compute_lambda = False

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 100
NEAT_CONFIG.elitism = 5

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

FITNESS_F = fitness_f
PAIR_SELECTION_F = sigma_scaled

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')

    DB_PATH = 'postgresql+psycopg2:///{}_{}'.format(PROBLEM_NAME, datetime.now().isoformat())

    DESCRIPTION = '"Majority problem"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(100):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
