import os
from statistics import mean

import seaborn
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from neat.nn import create_feed_forward_phenotype

from ca.iterate import ca_develop
from database import Db, Individual
from geometry.cell_grid import CellGrid2D, FiniteCellGrid2D
from patterns.replicate_pattern import find_pattern_partial_matches
from visualization.network_fig import draw_net

INTERVAL = 5


def get_db(path):
    return Db(path, echo=False)


if __name__ == '__main__':
    seaborn.set(style='white')

    from problems.replicate_twocolor import CA_CONFIG

    file = 'problems/results/' + 'replicate_twocolor/' + '2016-11-07 23:33:03.662880.db'
    db_path = 'sqlite:///{}'.format(file)
    generation_n = 50
    individual_n = 48

    db = get_db(db_path)
    session = db.Session()
    scenario = db.get_scenario(1, session=session)
    individual = session.query(Individual).filter(
        Individual.scenario_id == scenario.id,
        Individual.generation == generation_n,
        Individual.individual_number == individual_n,
    ).one()

    genotype = individual.genotype
    phenotype = create_feed_forward_phenotype(genotype)

    pattern = CA_CONFIG.etc['pattern']
    initial_grid = CellGrid2D(
        cell_states=CA_CONFIG.alphabet,
        neighbourhood=CA_CONFIG.neighbourhood,
    )
    initial_grid.add_pattern_at_coord(pattern, (0, 0))

    # CA_CONFIG.iterations = 24

    grid_iterations = tuple(ca_develop(phenotype, CA_CONFIG, initial_grid))

    for i, grid in enumerate(grid_iterations):
        if i == 0:
            continue

        partial_matches = find_pattern_partial_matches(grid, pattern=CA_CONFIG.etc['pattern'])
        wanted_occurences = 3
        print(i, mean(partial_matches) * min(1.0, len(partial_matches) / wanted_occurences), partial_matches)

    r = 50
    x_range = y_range = (-r, r)

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
    output_path = '{}_gen{}_ind{}'.format(os.path.basename(file).replace('.db', ''), generation_n, individual_n)
    # anim.save(output_path + '.gif', writer='imagemagick', fps=1)
    plt.show()

    print(genotype)
    # draw_net(genotype, filename=output_path + '.svg', view=True)
