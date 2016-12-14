from typing import T, Tuple

import os
import seaborn
from math import sqrt
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype
from operator import itemgetter

from ca.iterate import n_iterations
from database import Individual, get_db
from geometry.cell_grid import ToroidalCellGrid2D, get_all_rotations_and_flips, get_rotational_hash, CellGrid2D
from patterns.replicate_pattern import find_pattern_partial_matches
from utils import PROJECT_ROOT, create_state_normalization_rules
from visualization.colors import colormap, norm
from visualization.network_fig import draw_net

seen = set()

if __name__ == '__main__':
    seaborn.set(style='white')

    from problems.replicate_tricolor import CA_CONFIG

    problem_name, db_file = ('replicate_tricolor', '2016-11-21 17:29:01.273816.db')

    db_path = 'sqlite:///{}'.format(os.path.join(PROJECT_ROOT, 'problems', 'results', problem_name, db_file))
    db = get_db(db_path)
    session = db.Session()

    count = 0
    q = session.query(Individual).filter(Individual.fitness >= 1.0) \
        .order_by(Individual.scenario_id, Individual.generation, Individual.individual_number)
    print(q.count(), 'total')
    for individual in q:

        genotype = individual.genotype
        phenotype = create_feed_forward_phenotype(genotype)
        alphabet = CA_CONFIG.alphabet
        iterations = CA_CONFIG.iterations
        neighbourhood = CA_CONFIG.neighbourhood

        pattern = CA_CONFIG.etc['pattern']
        wanted_occurrences = CA_CONFIG.etc['wanted_occurrences']

        initial_grid = CellGrid2D(
            cell_states=alphabet,
            neighbourhood=neighbourhood,
        )
        initial_grid.add_pattern_at_coord(pattern, (0, 0))

        state_normalization_rules = create_state_normalization_rules(states=alphabet)


        def ca_develop(network: FeedForwardNetwork):
            def transition_f(inputs_discrete_values: Tuple[T]) -> T:
                if all((x == initial_grid.dead_cell) for x in inputs_discrete_values):
                    return initial_grid.dead_cell

                inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

                outputs = network.serial_activate(inputs_float_values)

                return max(zip(alphabet, outputs), key=itemgetter(1))[0]

            yield initial_grid

            for grid in n_iterations(initial_grid, transition_f, iterations):

                partial_matches = list(find_pattern_partial_matches(grid, pattern))

                best_n_matches = sorted(partial_matches, reverse=True)[:wanted_occurrences]

                yield grid

                if best_n_matches == [1.0, 1.0, 1.0]:
                    return


        grid_iterations = tuple(ca_develop(phenotype))

        print('show', individual.scenario_id, individual.generation, individual.individual_number)
        fig = plt.figure()

        grid_iterations_ = grid_iterations[:30]
        last = grid_iterations_[-1]
        (l, t), (r, b) = last.get_extreme_coords()
        extent = (l, r, b, t)
        x_range = (l - 1, r + 2)
        y_range = (t - 1, b + 2)

        for i, iteration in enumerate(grid_iterations_, 1):
            n = len(grid_iterations)
            ax = fig.add_subplot(n // 5 + 1, 5, i)
            plt.axis('off')
            ax.set_title(i - 1)
            im = plt.imshow(
                iteration.get_enumerated_rectangle(x_range=x_range, y_range=y_range),
                extent=extent,
                interpolation='none',
                cmap=colormap,
                norm=norm,
            )

    plt.show()
