import os
from statistics import mean, median, mode

import matplotlib.pyplot as plt
import seaborn
from neat.nn import create_feed_forward_phenotype
from sqlalchemy.sql.functions import func

from ca_neat.database import Db, Individual, Scenario, get_db
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.problems.majority import NEAT_CONFIG, CA_CONFIG, fitness_f, create_binary_pattern
from ca_neat.utils import PROJECT_ROOT, invert_pattern
from ca_neat.visualization.network_fig import draw_net

seaborn.set_style('white')

if __name__ == '__main__':

    DB_PATH = 'postgresql+psycopg2:///majority_2017-03-24T15:48:10.659623'

    print(DB_PATH)
    db = get_db(DB_PATH)

    s = db.Session()

    Xs = s.query(Individual) \
        .order_by(Individual.scenario_id, -Individual.fitness) \
        .distinct(Individual.scenario_id)

    N = 149

    CA_CONFIG.iterations = N
    CA_CONFIG.compute_lambda = False
    patterns = [create_binary_pattern(alphabet=CA_CONFIG.alphabet, length=N, r=0.5) for _ in range(50)]
    patterns.extend(list(map(lambda pattern: invert_pattern(pattern, alphabet=CA_CONFIG.alphabet), patterns)))
    CA_CONFIG.etc = {
        'test_patterns': [(pattern, mode(pattern)) for pattern in patterns]
    }

    for n, x in enumerate(Xs):
        gt = deserialize_gt(x.genotype, NEAT_CONFIG)
        title = 'run {} gen {} ind {}'.format(x.scenario_id, x.generation, x.individual_number)
        # draw_net(gt,
        #          title=title,
        #          filename=title,
        #          view=True,
        #          in_labels=['-3', '-2', '-1', '0', '1', '2', '3'],
        #          out_labels={
        #              i: v for i, v in enumerate(CA_CONFIG.alphabet, start=7)
        #              })

        f = fitness_f(create_feed_forward_phenotype(gt), CA_CONFIG)

        print(x, f)
        #input()
