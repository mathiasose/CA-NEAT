import os
from operator import itemgetter
from typing import Iterator, Sequence

import seaborn
from matplotlib import pyplot as plt
from matplotlib import animation
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype

from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found, iterate_ca_once_with_coord_inputs
from ca_neat.database import Individual, get_db
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.geometry.cell_grid import CELL_STATE_T, ToroidalCellGrid2D
from ca_neat.geometry.neighbourhoods import radius_2d
from ca_neat.utils import PROJECT_ROOT, create_state_normalization_rules
from ca_neat.visualization.colors import colormap, norm
from ca_neat.visualization.network_fig import draw_net

INTERVAL = 5

if __name__ == '__main__':
    seaborn.set(style='white')

    from ca_neat.problems.novelty.generate_border_find_innovations import CA_CONFIG, NEAT_CONFIG

    scenario_id = 1
    generation_n = 12
    individual_n = 44

    DB_PATH = 'postgresql+psycopg2:///generate_border_find_innovations_2017-04-21T20:14:55.212484'
    db = get_db(DB_PATH)
    session = db.Session()
    scenario = db.get_scenario(scenario_id, session=session)
    individual = session.query(Individual).filter(
        Individual.scenario_id == scenario.id,
        Individual.generation == generation_n,
        Individual.individual_number == individual_n,
    ).one()

    genotype = deserialize_gt(individual.genotype, neat_config=NEAT_CONFIG)
    phenotype = create_feed_forward_phenotype(genotype)
    alphabet = CA_CONFIG.alphabet
    iterations = CA_CONFIG.iterations
    neighbourhood = CA_CONFIG.neighbourhood

    # pattern = CA_CONFIG.etc['seed']
    # pattern_h, pattern_w = len(pattern), len(pattern[0])
    # x_range = (0, pattern_w)
    # y_range = (0, pattern_h)
    # initial_grid = ToroidalCellGrid2D(
    #     cell_states=alphabet,
    #     neighbourhood=neighbourhood,
    #     x_range=x_range,
    #     y_range=y_range,
    # )
    # initial_grid.add_pattern_at_coord(pattern, (0, 0))
    #
    # state_normalization_rules = create_state_normalization_rules(states=alphabet)
    # r = radius_2d(neighbourhood)
    # coord_normalization_rules = create_state_normalization_rules(states=range(0 - r, max(pattern_h, pattern_w) + r + 1))


    # def ca_develop(network: FeedForwardNetwork) -> Iterator[ToroidalCellGrid2D]:
    #     def transition_f(inputs_discrete_values: Sequence[CELL_STATE_T]) -> CELL_STATE_T:
    #         neighbour_values, xy_values = inputs_discrete_values[:-2], inputs_discrete_values[-2:]
    #
    #         if all((x == initial_grid.dead_cell) for x in neighbour_values):
    #             return initial_grid.dead_cell
    #
    #         inputs_float_values = tuple(state_normalization_rules[n] for n in neighbour_values) + \
    #                               tuple(coord_normalization_rules[n] for n in xy_values)
    #
    #         outputs = network.serial_activate(inputs_float_values)
    #
    #         return max(zip(alphabet, outputs), key=itemgetter(1))[0]
    #
    #     yield initial_grid
    #
    #     for grid in iterate_ca_n_times_or_until_cycle_found(
    #             initial_grid=initial_grid,
    #             transition_f=transition_f,
    #             n=iterations,
    #             iterate_f=iterate_ca_once_with_coord_inputs
    #     ):
    #         yield grid


    # grid_iterations = tuple(ca_develop(phenotype))
    #
    # for it in grid_iterations:
    #     print(it)
    #
    # fig = plt.figure()
    #
    # (l, r), (t, b) = x_range, y_range
    # extent = (l, r, b, t)
    #
    # im = plt.imshow(
    #     initial_grid.get_enumerated_rectangle(x_range=x_range, y_range=y_range),
    #     extent=extent,
    #     interpolation='none',
    #     cmap=colormap,
    #     norm=norm,
    # )
    #
    #
    # def init():
    #     im.set_data(initial_grid.get_enumerated_rectangle(x_range=x_range, y_range=y_range))
    #     # plt.title('{}/{}'.format(0, ca_config.iterations))
    #     return (im,)
    #
    #
    # def animate(i):
    #     im.set_array(grid_iterations[i].get_enumerated_rectangle(x_range=x_range, y_range=y_range))
    #     # plt.suptitle('{}/{}'.format(i, ca_config.iterations))
    #     return (im,)


    file_descriptor = 'TODO'
    output_path = 'run{}_gen{}_ind{}'.format(scenario_id, generation_n, individual_n)

    # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(grid_iterations), interval=500, blit=True)
    # anim.save(output_path + '.gif', writer='imagemagick', fps=1)
    # plt.show()

    # print(genotype)
    y_ = {0: 'N', 1: 'W', 2: 'p', 3: 'E', 4: 'S', 5: 'X', 6: 'Y'}
    d = dict((k, v) for k, v in
             (zip((k for (k, v) in genotype.node_genes.items() if v.type == 'OUTPUT'), CA_CONFIG.alphabet)))
    draw_net(genotype, filename=output_path + '_full', fmt='svg', view=True, in_labels=y_, out_labels=d,
             show_disabled=True, prune_unused=False)
    draw_net(genotype, filename=output_path + '_pruned', fmt='svg', view=True, in_labels=y_, out_labels=d,
             show_disabled=False, prune_unused=True)
