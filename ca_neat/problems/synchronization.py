import os
from datetime import datetime
from random import uniform

from neat.nn import FeedForwardNetwork

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import LLLCRRR
from ca_neat.patterns.patterns import ALPHABET_2
from ca_neat.problems.majority import create_binary_pattern
from ca_neat.run_experiment import initialize_scenario

P = 200
N = 49
M = 200
p = 1.0 / 2.0
E = 5
G = 50
K = 100
I = 20

assert K % 2 == 0


def fitness_f(phenotype: FeedForwardNetwork, ca_config: CAConfig) -> float:
    from operator import itemgetter
    from typing import Iterator, Sequence
    from neat.nn import FeedForwardNetwork
    from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
    from ca_neat.utils import create_state_normalization_rules
    from ca_neat.geometry.cell_grid import ToroidalCellGrid1D, CELL_STATE_T
    from random import shuffle

    GRID_T = ToroidalCellGrid1D

    alphabet = ca_config.alphabet

    iterations = ca_config.iterations
    state_normalization_rules = create_state_normalization_rules(states=alphabet)

    def ca_develop(grid, network: FeedForwardNetwork) -> Iterator[GRID_T]:
        def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield grid

        for grid_it in iterate_ca_n_times_or_until_cycle_found(grid, transition_f, iterations):
            yield grid_it

    def test(selection) -> int:
        points = 0
        for pattern in selection:
            x_range = (0, len(pattern))
            neighbourhood = ca_config.neighbourhood

            init_grid = GRID_T(
                cell_states=alphabet,
                x_range=x_range,
                values={(i,): x for i, x in enumerate(pattern)},
                neighbourhood=neighbourhood,
            )

            *_, second_last, last = ca_develop(init_grid, phenotype)

            a, b = next(last.yield_values()), next(second_last.yield_values())

            if a != b and all(x == a for x in last.yield_values()) and all(x == b for x in second_last.yield_values()):
                points += 1

        return points

    # Out of K total patterns we first do a test on I samples.
    # If any points are achieved from the I patterns, the rest of the patterns are also tested.
    # This prevents wasting too much computing time on static or chaotic rules that don't lead to anything useful

    patterns = ca_config.etc['test_patterns']
    K = len(patterns)
    I = ca_config.etc['I']
    shuffle(patterns)
    selection, rest = patterns[:I], patterns[I:]

    sample_points = test(selection)

    if sample_points == 0:
        return 0.0
    else:
        rest_points = test(rest)
        return (sample_points + rest_points) / K


CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_2
CA_CONFIG.neighbourhood = LLLCRRR
CA_CONFIG.iterations = M
CA_CONFIG.compute_lambda = False

patterns = []
for _ in range(K):
    while True:
        p = create_binary_pattern(alphabet=CA_CONFIG.alphabet, length=N, r=uniform(0, 1))

        if p in patterns:
            continue
        else:
            patterns.append(p)
            break

patterns = sorted(patterns, key=lambda xs: sum(x == ' ' for x in xs))

CA_CONFIG.etc = {
    'test_patterns': patterns,
    'I': I,
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = P
NEAT_CONFIG.generations = G
NEAT_CONFIG.elitism = E

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

FITNESS_F = fitness_f
PAIR_SELECTION_F = sigma_scaled

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')

    DB_PATH = 'postgresql+psycopg2:///synchronization_2017-05-06T17:00:23.012278'

    if DB_PATH:
        blob = os.path.join(os.path.dirname(THIS_FILE), 'synchronization_2017-05-06T17:00:23.012278.json')
        with open(blob, 'r') as f:
            import json

            CA_CONFIG.etc = json.load(f)
    else:
        dt = datetime.now().isoformat()
        DB_PATH = 'postgresql+psycopg2:///' + '{}_{}'.format(PROBLEM_NAME, dt)

        blob = os.path.join(os.path.dirname(THIS_FILE), '{}_{}.json'.format(PROBLEM_NAME, dt))
        with open(blob, 'w+') as f:
            import json

            f.write(json.dumps(CA_CONFIG.etc))

    DESCRIPTION = '"Synchronization problem"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(25):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )

"""
postgresql+psycopg2:///synchronization_2017-05-06T02:18:24.967136
    P = 200
    N = 49
    M = 200
    p = binomial
    E = 5
    G = 100
    K = 100
    I = 25


postgresql+psycopg2:///synchronization_2017-05-06T17:00:23.012278
    P = 200
    N = 49
    M = 200
    p = uniform
    E = 5
    G = 10
    K = 100
    I = 20
"""
