import os
from itertools import groupby

import matplotlib.pyplot as plt
import seaborn

from database import Db, Individual, Scenario, get_db
from utils import PROJECT_ROOT

seaborn.set_style('white')


def plot_generations_until_optimal(db: Db):
    session = db.Session()
    scenarios = db.get_scenarios(session=session)

    assert scenarios.count() >= 1

    plt.ion()
    ax = plt.gca()
    ax.set_color_cycle(seaborn.color_palette('hls', n_colors=scenarios.count()))
    plt.title('Generations until optimal solution found, cumulative histogram\n{} scenarios'.format(scenarios.count()))

    q = session.query(Scenario, Individual.generation) \
        .join(Individual) \
        .filter(Individual.fitness == 1.0) \
        .group_by(Scenario.id) \
        .distinct(Scenario.id) \
        .order_by(Individual.generation)

    print('{}/{}'.format(q.count(), scenarios.count()))
    gens = [r[1] for r in q.all()]

    for n, group in groupby(gens, lambda x: x):
        print(n, '-' * len(tuple(group)))

    n = max(gens)
    ax.hist(gens, bins=n, range=(0, n), cumulative=True)
    plt.show(block=True)

    return plt


if __name__ == '__main__':
    problem_dir = 'replicate_mosaic/'
    db_file = '2016-11-16 20:49:47.478567.db'

    THIS_FILE = os.path.abspath(__file__)
    RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'problems', 'results'))
    file = os.path.join(RESULTS_DIR, problem_dir, db_file)
    db_path = 'sqlite:///{}'.format(file)

    print(db_path)
    db = get_db(db_path)
    plot_generations_until_optimal(db)
