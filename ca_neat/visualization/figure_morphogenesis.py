import os
from operator import itemgetter
from typing import T, Tuple, Sequence, Iterator

import seaborn
from matplotlib import pyplot as plt, rcParams
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype

from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found, iterate_ca_once_with_coord_inputs
from ca_neat.database import Individual, get_db
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.geometry.cell_grid import ToroidalCellGrid2D, get_rotational_hash, CELL_STATE_T
from ca_neat.geometry.neighbourhoods import radius_2d
from ca_neat.utils import PROJECT_ROOT, create_state_normalization_rules
from ca_neat.visualization.colors import colormap, norm

rcParams['savefig.format'] = 'pdf'

seen = set()

if __name__ == '__main__':
    seaborn.set(style='white')

    from ca_neat.problems.novelty.generate_swiss_use_innovations import CA_CONFIG, NEAT_CONFIG

    DB_PATH = 'postgresql+psycopg2:///generate_swiss_use_innovations_2017-06-11T14:41:05.471556'
    DB = get_db(DB_PATH)
    session = DB.Session()

    count = 0
    q = session.query(Individual) \
        .order_by(-Individual.fitness)

    print(q.count(), 'total')

    for individual in q[:10]:
        genotype = deserialize_gt(individual.genotype, NEAT_CONFIG)
        phenotype = create_feed_forward_phenotype(genotype)
        alphabet = CA_CONFIG.alphabet
        iterations = CA_CONFIG.iterations
        neighbourhood = CA_CONFIG.neighbourhood

        pattern = CA_CONFIG.etc['seed']
        pattern_h, pattern_w = len(pattern), len(pattern[0])
        x_range = (0, pattern_w)
        y_range = (0, pattern_h)
        initial_grid = ToroidalCellGrid2D(
            cell_states=alphabet,
            neighbourhood=neighbourhood,
            x_range=x_range,
            y_range=y_range,
        )
        initial_grid.add_pattern_at_coord(pattern, (0, 0))

        state_normalization_rules = create_state_normalization_rules(states=alphabet)
        r = radius_2d(neighbourhood)
        coord_normalization_rules = create_state_normalization_rules(
            states=range(0 - r, max(pattern_h, pattern_w) + r + 1))


        def ca_develop(network: FeedForwardNetwork) -> Iterator[ToroidalCellGrid2D]:
            def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
                if all((x == initial_grid.dead_cell) for x in inputs_discrete_values):
                    return initial_grid.dead_cell

                inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

                outputs = network.serial_activate(inputs_float_values)

                return max(zip(alphabet, outputs), key=itemgetter(1))[0]

            yield initial_grid

            for grid in iterate_ca_n_times_or_until_cycle_found(initial_grid, transition_f, iterations):
                yield grid


        grid_iterations = tuple(ca_develop(phenotype))

        iterations_hash = hash(sum(get_rotational_hash(x) for x in grid_iterations))

        if iterations_hash in seen:
            print('skip', individual.scenario_id, individual.generation, individual.individual_number)
            continue
        else:
            seen.add(iterations_hash)
            count += 1

        print('show', individual.scenario_id, individual.generation, individual.individual_number)
        fig = plt.figure()
        fig.canvas.set_window_title(
            'run {} gen {} ind {}'.format(individual.scenario_id, individual.generation, individual.individual_number))
        plt.axis('off')
        # plt.title('\t'.join(map(str, [individual.scenario_id, individual.generation, individual.individual_number])))

        (l, r), (t, b) = x_range, y_range
        extent = (l, r, b, t)

        asdf = dict()
        for i, iteration in enumerate(grid_iterations[:30], 1):
            n = len(grid_iterations)
            ax = fig.add_subplot(n // 5 + 1, 5, i)
            ax.set_title(asdf.get(iteration.__hash__(), i - 1))
            plt.axis('off')
            im = plt.imshow(
                iteration.get_enumerated_rectangle(x_range=x_range, y_range=y_range),
                extent=extent,
                interpolation='none',
                cmap=colormap,
                norm=norm,
            )
            asdf[iteration.__hash__()] = i - 1

    plt.show()
    print(count, 'unique')

"""16 2 182"""
