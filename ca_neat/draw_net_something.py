from ca_neat.database import get_db, Individual
from ca_neat.ca.iterate import iterate_ca_once_with_coord_inputs, iterate_ca_n_times_or_until_cycle_found
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.patterns.replicate_pattern import count_correct_cells
from ca_neat.geometry.cell_grid import ToroidalCellGrid2D
from ca_neat.utils import create_state_normalization_rules
from operator import itemgetter
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype
from math import exp
from typing import Sequence, Iterator
from ca_neat.geometry.cell_grid import CELL_STATE_T
from ca_neat.geometry.neighbourhoods import radius_2d
from ca_neat.visualization.network_fig import draw_net

from ca_neat.problems.synchronization import CA_CONFIG, NEAT_CONFIG

# neighbourhood = CA_CONFIG.neighbourhood
alphabet = CA_CONFIG.alphabet
# target_pattern = CA_CONFIG.etc['target_pattern']
# seed = CA_CONFIG.etc['seed']
# iterations = CA_CONFIG.iterations
#
# pattern_w = len(target_pattern[0])
# pattern_h = len(target_pattern)
#
# initial_grid = ToroidalCellGrid2D(
#     cell_states=alphabet,
#     neighbourhood=neighbourhood,
#     x_range=(0, pattern_w),
#     y_range=(0, pattern_h),
# )
# DEAD_CELL = initial_grid.dead_cell
# initial_grid.add_pattern_at_coord(seed, (0, 0))
# state_normalization_rules = create_state_normalization_rules(states=alphabet)
# r = radius_2d(neighbourhood)
# coord_normalization_rules = create_state_normalization_rules(states=range(0 - r, max(pattern_h, pattern_w) + r + 1))
#
#
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


if __name__ == '__main__':

    DB_PATH = 'postgresql+psycopg2:///generate_swiss_use_innovations_2017-06-11T14:41:05.471556'
    print(DB_PATH)
    db = get_db(DB_PATH)
    s = db.Session()

    Xs = s.query(Individual) \
        .order_by(-Individual.fitness)[:10]
    Ss = []

    for n, x in enumerate(Xs):
        gt = deserialize_gt(x.genotype, NEAT_CONFIG)
        title = 'run {} gen {} ind {}'.format(x.scenario_id, x.generation, x.individual_number)
        draw_net(gt,
                 title=title,
                 filename=title,
                 view=True,
                 in_labels=['L3', 'L2', 'L1', 'C', 'R1', 'R2', 'R3'],
                 out_labels={
                     i: v for i, v in enumerate(alphabet, start=7)
                     })
        #     pt = create_feed_forward_phenotype(gt)
        #     grid_iterations = list(ca_develop(pt))
        #
        #     best = 0.0
        #     best_i = 0
        #     for i, grid in enumerate(grid_iterations):
        #         correctness_fraction = count_correct_cells(grid.get_whole(), target_pattern=target_pattern) / grid.area
        #
        #         if correctness_fraction > best:
        #             best = correctness_fraction
        #             best_i = i
        #
        #     Ss.append(best)
        #
        #     print(x, best)
        #     print(grid_iterations[best_i])
        #
        # print(sorted(Ss))
