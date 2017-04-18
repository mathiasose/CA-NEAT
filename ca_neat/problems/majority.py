import os
from datetime import datetime
from statistics import mode
from typing import Sequence, Tuple

from neat.nn import FeedForwardNetwork

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.cell_grid import CELL_STATE_T
from ca_neat.geometry.neighbourhoods import LLLCRRR
from ca_neat.patterns.patterns import ALPHABET_2
from ca_neat.run_experiment import initialize_scenario
from ca_neat.utils import invert_pattern, random_string

N = 49


def create_binary_pattern(alphabet: Tuple[CELL_STATE_T, CELL_STATE_T], length=N, r=1.0 / 3.0) -> Sequence[CELL_STATE_T]:
    from random import choice, shuffle

    a, b = alphabet

    r = int(r * length)

    values = [a] * r + [b] * (length - r)

    while len(values) < length:
        values.append(choice(alphabet))

    shuffle(values)

    return values


def fitness_f(phenotype: FeedForwardNetwork, ca_config: CAConfig) -> float:
    from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
    from ca_neat.utils import create_state_normalization_rules
    from operator import itemgetter
    from neat.nn import FeedForwardNetwork
    from typing import Iterator
    from statistics import mean
    from ca_neat.geometry.cell_grid import ToroidalCellGrid1D
    from math import exp
    from ca_neat.geometry.cell_grid import CELL_STATE_T
    from typing import Sequence

    alphabet = ca_config.alphabet

    iterations = ca_config.iterations
    state_normalization_rules = create_state_normalization_rules(states=alphabet)

    def ca_develop(grid, network: FeedForwardNetwork) -> Iterator[ToroidalCellGrid1D]:
        def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
            if all((x == grid.dead_cell) for x in inputs_discrete_values):
                return grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield grid

        for grid_it in iterate_ca_n_times_or_until_cycle_found(grid, transition_f, iterations):
            yield grid_it

    k = 1
    redistribute = lambda x: x * exp(k * x) / exp(k)

    results = []
    for pattern, majority in ca_config.etc['test_patterns']:
        x_range = (0, len(pattern))
        neighbourhood = ca_config.neighbourhood

        init_grid = ToroidalCellGrid1D(
            cell_states=alphabet,
            x_range=x_range,
            values={(i,): x for i, x in enumerate(pattern)},
            neighbourhood=neighbourhood,
        )

        grid_iterations = tuple(ca_develop(init_grid, phenotype))
        values = grid_iterations[-1].get_whole()

        results.append(
            sum(x == majority for x in values) / init_grid.area
        )

    return redistribute(mean(results))


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_2
CA_CONFIG.neighbourhood = LLLCRRR
CA_CONFIG.iterations = N
CA_CONFIG.compute_lambda = False

patterns = [create_binary_pattern(alphabet=CA_CONFIG.alphabet, length=N) for _ in range(5)]
patterns.extend(list(map(lambda pattern: invert_pattern(pattern, alphabet=CA_CONFIG.alphabet), patterns)))
CA_CONFIG.etc = {
    'test_patterns': [(pattern, mode(pattern)) for pattern in patterns]
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 100
NEAT_CONFIG.elitism = 40

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

FITNESS_F = fitness_f
PAIR_SELECTION_F = sigma_scaled

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'results', PROBLEM_NAME))
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    DB_PATH = 'postgresql+psycopg2:///' + '{}_{}'.format(PROBLEM_NAME, datetime.now().isoformat())

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
