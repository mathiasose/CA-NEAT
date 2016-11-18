import os

import matplotlib.pyplot as plt
import seaborn
from sqlalchemy.sql.functions import func

from database import Db, Individual, get_db
from utils import PROJECT_ROOT

seaborn.set_style('white')


def plot_fitnesses_over_generations(db: Db):
    session = db.Session()
    scenarios = db.get_scenarios(session=session)

    assert scenarios

    for scenario in scenarios:
        inds = db.get_individuals(scenario_id=scenario.id)
        count = inds.count()
        if count == 0:
            print(scenario.id, 0, 0)
            continue

        maxx = max(i.fitness for i in inds)

        if maxx == 1.0:
            continue

        print(scenario.id, maxx, count)

    exit(0)

    plt.ion()
    plt.gca().set_color_cycle(seaborn.color_palette('hls', n_colors=scenarios.count()))

    for i, scenario in enumerate(scenarios):
        progression = session \
            .query(Individual, func.max(Individual.fitness)) \
            .filter(Individual.scenario_id == scenario.id) \
            .order_by(Individual.generation) \
            .group_by(Individual.generation) \
            .order_by(func.max(Individual.fitness).desc())
        plt.plot([score for _, score in progression])
        print(i, progression.count())

    plt.show(block=True)

    return plt


if __name__ == '__main__':
    problem_dir = 'replicate_swiss_flag/'
    db_file = '2016-11-16 12:39:05.554163.db'

    THIS_FILE = os.path.abspath(__file__)
    RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'problems', 'results'))
    file = os.path.join(RESULTS_DIR, problem_dir, db_file)
    db_path = 'sqlite:///{}'.format(file)

    print(db_path)
    db = get_db(db_path)
    plot_fitnesses_over_generations(db)
