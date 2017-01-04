import os
from statistics import mean, median

import matplotlib.pyplot as plt
import seaborn
from sqlalchemy.sql.functions import func

from database import Db, Individual, Scenario, get_db
from utils import PROJECT_ROOT

seaborn.set_style('white')


def plot(db: Db, show=True, n=None):
    session = db.Session()
    scenarios = db.get_scenarios(session=session)

    assert scenarios.count()

    scenarios_fitnesses = []
    for scenario in scenarios:
        progression = session \
            .query(Individual, func.max(Individual.fitness)) \
            .filter(Individual.scenario_id == scenario.id) \
            .order_by(Individual.generation) \
            .group_by(Individual.generation) \
            .order_by(func.max(Individual.fitness).desc())
        print(scenario.id, progression.count(), any(f >= 1.0 for _, f in progression))
        scenarios_fitnesses.append([score for _, score in progression])

    n = n or max(map(len, scenarios_fitnesses))
    for sf in scenarios_fitnesses:
        while len(sf) < n:
            sf.append(sf[-1])

    q = session.query(Scenario, Individual.generation) \
        .join(Individual) \
        .filter(Individual.fitness == 1.0) \
        .group_by(Scenario.id) \
        .distinct(Scenario.id) \
        .order_by(Individual.generation)

    gens = [r[1] for r in q.all()]

    if show:
        plt.ion()

    plt.title(scenarios[0].description.split('\n')[0])
    plt.xlabel('Generations')

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.set_zorder(200)
    ax2.set_zorder(100)
    ax1.patch.set_visible(False)

    ax1.set_color_cycle(seaborn.color_palette('hls', n_colors=scenarios.count()))
    ax1.axis([0, n, -0.1, 1.1])
    ax1.set_ylabel('Fitness')

    ax2.set_ylabel('Finished runs')
    ax2.set_ylim([0, int(scenarios.count() * 1.1)])

    ax1.plot([mean(gen) for gen in zip(*scenarios_fitnesses)], 'r', label='mean', zorder=210)
    ax1.plot([median(gen) for gen in zip(*scenarios_fitnesses)], 'b--', label='median', zorder=211)

    ax2.hist(gens, bins=n, range=(0, n), cumulative=True, color=['pink'], zorder=110, label='Finished runs')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    l = plt.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc='lower right')
    l.set_zorder(300)

    if show:
        plt.show(block=True)
    else:
        filename = input('Filename?\t').lower()
        plt.savefig(filename)

    return plt


if __name__ == '__main__':
    problem_dir, db_file = ('generate_swiss_flag', '2016-11-18 21:48:15.168123.db')

    THIS_FILE = os.path.abspath(__file__)
    RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'problems', 'results'))
    file = os.path.join(RESULTS_DIR, problem_dir, db_file)
    db_path = 'sqlite:///{}'.format(file)

    print(db_path)
    db = get_db(db_path)
    plot(db, show=True)
