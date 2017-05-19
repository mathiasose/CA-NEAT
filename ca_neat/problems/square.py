import os
from datetime import datetime

from neat.nn import FeedForwardNetwork

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import LLLCRRR
from ca_neat.patterns.patterns import ALPHABET_2
from ca_neat.run_experiment import initialize_scenario

STATES = 8
SIZE = 100
ITERATIONS = 100

TESTS = list(range(2, 7))


def fitness_f(phenotype: FeedForwardNetwork, ca_config: CAConfig) -> float:
    from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
    from ca_neat.utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from typing import Iterator
    from ca_neat.geometry.cell_grid import CellGrid1D
    from ca_neat.geometry.cell_grid import CELL_STATE_T
    from typing import Sequence

    GRID_T = CellGrid1D

    alphabet = ca_config.alphabet

    iterations = ca_config.iterations
    state_normalization_rules = create_state_normalization_rules(states=alphabet)

    def ca_develop(grid, network: FeedForwardNetwork) -> Iterator[GRID_T]:
        def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
            if all((x == grid.dead_cell) for x in inputs_discrete_values):
                return grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield grid

        for grid_it in iterate_ca_n_times_or_until_cycle_found(grid, transition_f, iterations):
            yield grid_it

    tests = ca_config.etc['tests']
    points = 0

    for test_n in tests:
        correct = test_n ** 2

        init_grid = GRID_T(
            cell_states=alphabet,
            values={(x,): 1 for x in range(SIZE // 2, SIZE // 2 + test_n)},
            neighbourhood=ca_config.neighbourhood,
        )

        *_, last = ca_develop(init_grid, phenotype)

        s = last.__str__().strip(init_grid.dead_cell)

        if not s:
            continue

        if len(s) == correct and all(x == s[0] for x in s):
            points += 1

    return points / len(tests)


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_2
CA_CONFIG.neighbourhood = LLLCRRR
CA_CONFIG.iterations = ITERATIONS
CA_CONFIG.compute_lambda = False

CA_CONFIG.etc = {
    'tests': TESTS,
    'grid_size': SIZE,
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 10
NEAT_CONFIG.elitism = 5

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

FITNESS_F = fitness_f
PAIR_SELECTION_F = sigma_scaled

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')

    dt = datetime.now().isoformat()

    DB_PATH = 'postgresql+psycopg2:///' + '{}_{}'.format(PROBLEM_NAME, dt)

    blob = os.path.join(os.path.dirname(THIS_FILE), '{}_{}.json'.format(PROBLEM_NAME, dt))
    with open(blob, 'w+') as f:
        import json

        f.write(json.dumps(CA_CONFIG.etc))

    DESCRIPTION = '"Square problem"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(1):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
