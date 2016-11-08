import os

import matplotlib.pyplot as plt
import seaborn
from matplotlib.colors import ListedColormap
from neat.nn import create_feed_forward_phenotype

from ca.iterate import ca_develop
from problems.checkerboard import CA_CONFIG
from run_experiment import get_db


def plot_grid(cell_grid):
    seaborn.set(style='white')

    (l, t), (r, b) = cell_grid.get_extreme_coords()
    extent = (l, r, b, t)

    whole = [[int(c) for c in row] for row in cell_grid.get_whole()]

    plt.imshow(
        whole,
        extent=extent,
        interpolation='nearest',
        cmap=ListedColormap(seaborn.color_palette('bright', n_colors=len(cell_grid.cell_states))),
    )
    plt.show()


if __name__ == '__main__':
    DB_PATH = os.path.join('sqlite:///', 'db/checkerboard/2016-10-25 13:46:20.805981.db')
    DB = get_db(DB_PATH)

    scenario = DB.get_scenario(scenario_id=1)
    gen = DB.get_generation(scenario_id=1, generation=(scenario.generations - 1))

    for generation, individual, fitness in DB.get_top_performing_for_generation(scenario_id=1):
        genotype = individual.genotype
        phenotype = create_feed_forward_phenotype(genotype)
        grid = ca_develop(phenotype, CA_CONFIG)
        plot_grid(grid)
