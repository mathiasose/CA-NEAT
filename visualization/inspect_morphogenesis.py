import os
from operator import itemgetter
from typing import T, Tuple

import seaborn
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import BoundaryNorm, ListedColormap
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype

from ca.iterate import iterate_ca_n_times_or_until_cycle_found
from database import Individual, get_db
from geometry.cell_grid import ToroidalCellGrid2D
from utils import PROJECT_ROOT, create_state_normalization_rules
from visualization.colors import colormap, norm
from visualization.network_fig import draw_net

INTERVAL = 5

if __name__ == '__main__':
    seaborn.set(style='white')

    from problems.generate_tricolor import CA_CONFIG

    problem_name, db_file = ('generate_tricolor', '2016-12-04 18:09:12.299816.db')
    scenario_id = 47
    generation_n = 301
    individual_n = 100

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


    def ca_develop(network: FeedForwardNetwork):
        def transition_f(inputs_discrete_values: Tuple[T]) -> T:
            if all((x == initial_grid.dead_cell) for x in inputs_discrete_values):
                return initial_grid.dead_cell

            inputs_float_values = tuple(state_normalization_rules[x] for x in inputs_discrete_values)

            outputs = network.serial_activate(inputs_float_values)

            return max(zip(alphabet, outputs), key=itemgetter(1))[0]

        yield initial_grid

        for grid in iterate_ca_n_times_or_until_cycle_found(initial_grid, transition_f, iterations):
            yield grid


    grid_iterations = tuple(ca_develop(phenotype))

    for it in grid_iterations:
        print(it)

    fig = plt.figure()

    (l, r), (t, b) = x_range, y_range
    extent = (l, r, b, t)

    im = plt.imshow(
        initial_grid.get_enumerated_rectangle(x_range=x_range, y_range=y_range),
        extent=extent,
        interpolation='none',
        cmap=colormap,
        norm=norm,
    )


    def init():
        im.set_data(initial_grid.get_enumerated_rectangle(x_range=x_range, y_range=y_range))
        # plt.title('{}/{}'.format(0, ca_config.iterations))
        return (im,)


    def animate(i):
        im.set_array(grid_iterations[i].get_enumerated_rectangle(x_range=x_range, y_range=y_range))
        # plt.suptitle('{}/{}'.format(i, ca_config.iterations))
        return (im,)


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(grid_iterations), interval=500, blit=True)
    file_descriptor = '{}_{}'.format(problem_name, db_file.replace('.db', ''))
    output_path = '{}_gen{}_ind{}'.format(file_descriptor, generation_n, individual_n)

    # anim.save(output_path + '.gif', writer='imagemagick', fps=1)
    plt.show()

    print(genotype)
    print(dict((k, v) for k, v in
               (zip((str(k) for (k, v) in genotype.node_genes.items() if v.type == 'OUTPUT'), CA_CONFIG.alphabet))))
    draw_net(genotype, filename=output_path, fmt='png', view=True)
