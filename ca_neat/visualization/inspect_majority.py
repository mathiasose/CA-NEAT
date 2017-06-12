from collections import defaultdict, Counter
from operator import itemgetter
from random import uniform
from statistics import mode, mean
from typing import Iterator
from typing import Sequence

from math import exp
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype
from sqlalchemy.sql.functions import func

from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found, iterate_ca_n_times
from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.geometry.cell_grid import CELL_STATE_T
from ca_neat.geometry.cell_grid import FiniteCellGrid1D, ToroidalCellGrid1D
from ca_neat.problems.majority import create_binary_pattern
from ca_neat.problems.synchronization import CA_CONFIG, NEAT_CONFIG
from ca_neat.utils import create_state_normalization_rules, invert_pattern, random_string

P = 200
N = 149
M = 500
p = 1.0 / 2.0
E = 5
G = 100
K = 100
I = 25

if __name__ == '__main__':
    DB_PATH = 'postgresql+psycopg2:///majority_2017-05-02T17:42:29.143748'
    db = get_db(DB_PATH)

    session = db.Session()

    assert session.query(Individual).count()

    individuals = session.query(Individual) \
        .filter(Individual.fitness == 1.0)

    assert individuals.count()

    individuals = individuals[:10]

    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_style('white')

    alphabet = CA_CONFIG.alphabet

    iterations = M
    state_normalization_rules = create_state_normalization_rules(states=alphabet)

    CA_CONFIG.etc['test_patterns'] = [create_binary_pattern(alphabet=alphabet, length=N, r=1 / 2.5) for _ in range(1)]


    # print(*(''.join(p) for p in CA_CONFIG.etc['test_patterns']), sep='\n')


    def ca_develop(grid, network: FeedForwardNetwork) -> Iterator[ToroidalCellGrid1D]:
        def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield grid

        for grid_it in iterate_ca_n_times_or_until_cycle_found(grid, transition_f, M):
            yield grid_it


    y = defaultdict(list)
    z = {}
    for iii, individual in enumerate(individuals):
        gt = deserialize_gt(individual.genotype, NEAT_CONFIG)
        pt = create_feed_forward_phenotype(gt)

        l = []
        x = []
        points = 0
        patterns = CA_CONFIG.etc['test_patterns']
        for i, pattern in enumerate(patterns):
            init_grid = ToroidalCellGrid1D(
                cell_states=alphabet,
                x_range=(0, len(pattern)),
                values={(i,): x for i, x in enumerate(pattern)},
                neighbourhood=CA_CONFIG.neighbourhood,
            )
            z[i] = sum(x != ' ' for x in pattern)

            grid_iterations = list(ca_develop(init_grid, pt))

            # print(*grid_iterations, sep='\n')
            # input('----')

            second_last = grid_iterations[-2].get_whole()
            last = grid_iterations[-1].get_whole()

            if all(x == mode(pattern) for x in last) and last == second_last:
                points += 1
                x.append('█')
                l.append(len(grid_iterations))

                y[i].append('█')

                f = lambda xs: '█' if xs != ' ' else xs
                print(
                    *(
                        (
                            str(i).zfill(len(str(len(grid_iterations)))), ''.join(it.get_whole(f))
                        ) for (i, it) in enumerate(grid_iterations)
                    ), sep='\n'
                )
                print('-' * 100)

                key = list(alphabet)
                W = sum(x == ' ' for x in pattern)
                B = sum(x == '■' for x in pattern)

                fig=plt.figure()
                fig.canvas.set_window_title('run {} gen {} ind {}'.format(
                    individual.scenario_id, individual.generation, individual.individual_number
                ))
                plt.title(f'Majority task\nInitial state: {W} white, {B} black')
                plt.imshow(
                    [
                        [key.index(x) for x in row.get_whole()] for row in grid_iterations],
                    interpolation='none'
                )
                plt.ylabel('Time')
                plt.xlabel('Cells')
            else:
                x.append(' ')
                y[i].append(' ')

                # print(repr(individual), points / len(patterns))
                # print(iii, '\t', individual.scenario_id, '\t', individual.generation, '\t', *x, '\t',
                #       points / len(patterns), '\t', mean(l), sep='')

    plt.show()

    print()
    for k, v in y.items():
        print(k, end='\t')
        print(sum(x == '█' for x in v), end='\t')
        print(*v, sep='', end='\t')
        print(z[k])
