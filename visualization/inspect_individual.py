import os
from operator import itemgetter
from statistics import mean
from typing import T, Tuple

import seaborn
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype

from ca.iterate import n_iterations
from database import Individual, get_db
from geometry.cell_grid import CellGrid2D, ToroidalCellGrid2D
from patterns.replicate_pattern import find_pattern_partial_matches
from utils import PROJECT_ROOT, create_state_normalization_rules
from visualization.network_fig import draw_net

INTERVAL = 5

if __name__ == '__main__':
    seaborn.set(style='white')

    from problems.generate_swiss_flag import CA_CONFIG

    problem_name = 'generate_swiss_flag'
    db_file = '2016-11-17 23:23:26.460816.db'
    scenario_id = 52
    individual_n = 118
    generation_n = 12

    db_path = 'sqlite:///{}'.format(os.path.join(PROJECT_ROOT, 'problems', 'results', problem_name, db_file))
    db = get_db(db_path)
    session = db.Session()
    scenario = db.get_scenario(scenario_id, session=session)
    individual = session.query(Individual).filter(
        Individual.scenario_id == scenario.id,
        Individual.generation == generation_n,
        Individual.individual_number == individual_n,
    ).one()

    genotype = individual.genotype
    phenotype = create_feed_forward_phenotype(genotype)
    alphabet = CA_CONFIG.alphabet
    iterations = CA_CONFIG.iterations
    neighbourhood = CA_CONFIG.neighbourhood

    pattern = CA_CONFIG.etc['initial_pattern']
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


    def ca_develop(network: FeedForwardNetwork):
        def transition_f(inputs_discrete_values: Tuple[T]) -> T:
            if all((x == initial_grid.dead_cell) for x in inputs_discrete_values):
                return initial_grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield initial_grid

        for grid in n_iterations(initial_grid, transition_f, iterations):
            yield grid


    grid_iterations = tuple(ca_develop(phenotype))


    fig = plt.figure()
    # ax = plt.axes(xlim=x_range, ylim=y_range)

    n_colors = len(CA_CONFIG.alphabet)
    if n_colors == 2:
        colormap = ListedColormap(seaborn.color_palette(['#FFFFFF', '#FF0000'], n_colors=n_colors))
    else:
        colormap = ListedColormap(seaborn.color_palette('colorblind', n_colors=n_colors))

    (l, r), (t, b) = x_range, y_range
    extent = (l, r, b, t)

    im = plt.imshow(
        initial_grid.get_enumerated_rectangle(x_range=x_range, y_range=y_range),
        extent=extent,
        interpolation='none',
        cmap=colormap
    )


    def init():
        im.set_data(initial_grid.get_enumerated_rectangle(x_range=x_range, y_range=y_range))
        # plt.title('{}/{}'.format(0, ca_config.iterations))
        return (im,)


    def animate(i):
        im.set_array(grid_iterations[i].get_enumerated_rectangle(x_range=x_range, y_range=y_range))
        # plt.suptitle('{}/{}'.format(i, ca_config.iterations))
        return (im,)


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(grid_iterations), interval=1000, blit=True)
    file_descriptor = '{}_{}'.format(problem_name, db_file.replace('.db', ''))
    output_path = '{}_gen{}_ind{}'.format(file_descriptor, generation_n, individual_n)
    # anim.save(output_path + '.gif', writer='imagemagick', fps=1)
    plt.show()

    exit(0)
    print(genotype)
    print(dict((k, v) for k, v in
               (zip((str(k) for (k, v) in genotype.node_genes.items() if v.type == 'OUTPUT'), CA_CONFIG.alphabet))))
    draw_net(genotype, filename=output_path, fmt='png', view=True)
