from collections import Counter
from itertools import groupby
from statistics import mean, median, mode, stdev, pstdev, pvariance, StatisticsError
from typing import T, Tuple

import os
import seaborn
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype
from operator import itemgetter

from tabulate import tabulate

from ca.iterate import n_iterations
from database import Individual, get_db
from geometry.cell_grid import ToroidalCellGrid2D
from utils import PROJECT_ROOT, create_state_normalization_rules
from visualization.network_fig import draw_net


def modes(xs):
    counter = Counter(xs)
    n = counter.most_common(1)[0][1]
    return sorted([x[0] for x in next(groupby(counter.most_common(), lambda x: x[1]))[1]]), n


if __name__ == '__main__':
    seaborn.set(style='white')

    table = []
    for problem_name, db_file in [
        ('generate_mosaic', '2016-12-05 15:01:33.095161.db'),
        ('generate_border', '2016-11-23 13:51:24.771958.db'),
        ('generate_tricolor', '2016-12-04 18:09:12.299816.db'),
        ('generate_swiss_flag', '2016-11-18 21:48:15.168123.db'),
        ('replicate_mosaic', '2016-11-21 01:36:38.266342.db'),
        ('replicate_swiss_flag', '2016-11-16 12:39:05.554163.db'),
        ('replicate_tricolor', '2016-11-21 17:29:01.273816.db'),
        #('replicate_norwegian_flag', '2016-11-25 14:55:53.727034.db'),
    ]:
        db_path = 'sqlite:///{}'.format(os.path.join(PROJECT_ROOT, 'problems', 'results', problem_name, db_file))
        db = get_db(db_path)
        session = db.Session()

        sizetups = [individual.genotype.size() for individual in
                    session.query(Individual).filter(Individual.fitness >= 1.0)]
        ncs = [nodes for nodes, conns in sizetups]
        ccs = [conns for nodes, conns in sizetups]
        acs = [nodes + conns for nodes, conns in sizetups]

        for cs, label in zip([ncs, ccs, acs, ], ['Nodes', 'Connections', 'Both']):
            s = '{} ({} results)'.format(problem_name, len(sizetups))
            modess, n = modes(cs)
            s1 = '{} ({} occurrences)'.format(', '.join(map(str, modess)), n)
            table.append([s, label, min(cs), max(cs), mean(cs), median(cs), s1, pstdev(cs)])
        table.append([])


        #     plt.figure()
        #     plt.ylabel('Occurrences')
        #     plt.xlabel('Number of hidden nodes')
        #     plt.title(problem_name)
        #     plt.hist([ncs])
        #
        # plt.show()

    print(
        tabulate(
            headers=['Min', 'Max', 'Mean', 'Median', 'Mode(s)', '$\sigma$'],
            tabular_data=table,
            tablefmt='latex',
        )
    )
