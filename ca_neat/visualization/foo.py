import os
from statistics import mean, median

import matplotlib.pyplot as plt
import seaborn
from sqlalchemy.sql.functions import func

from ca_neat.database import Db, Individual, Scenario, get_db
from ca_neat.utils import PROJECT_ROOT

seaborn.set_style('white')


def plot(db: Db, show=True, n=None):
    session = db.Session()

    scenarios = db.get_scenarios(session=session)
    assert scenarios.count()

    x = session.query(Individual) \
        .filter(Individual.fitness >= 1.0) \
        .order_by(Individual.scenario_id, Individual.generation) \
        .distinct(Individual.scenario_id)

    gens = [i.generation for i in x]

    n = scenarios[0].generations or max(gens)

    scenarios_fitnesses = []
    # for i, scenario in enumerate(scenarios, 1):
    #     sid = scenario.id
    #
    #     p = session.query(Individual) \
    #         .filter(Individual.scenario_id == scenario.id)
    #
    #     k = session.query(p.filter(Individual.fitness >= 1.0).exists()).scalar()
    #
    #     progression = p \
    #         .order_by(Individual.generation) \
    #         .group_by(Individual.generation, Individual.scenario_id, Individual.individual_number) \
    #         .order_by(func.max(Individual.fitness).desc())
    #
    #     print(('✓' if k else ' '), i, sid, progression.count(), sep='\t')
    #
    #     if not progression.count():
    #         continue

    #     scenarios_fitnesses.append([score for _, score in progression])
    #
    # n = n or max(map(len, scenarios_fitnesses))
    # for sf in scenarios_fitnesses:
    #     while len(sf) < n:
    #         sf.append(sf[-1])

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

    #ax1.plot([mean(gen) for gen in zip(*scenarios_fitnesses)], 'r', label='mean', zorder=210)
    #ax1.plot([median(gen) for gen in zip(*scenarios_fitnesses)], 'k--', label='median', zorder=211)

    ax2.hist(gens, bins=n, range=(0, n), cumulative=True, color=['pink'], zorder=110, label='Finished runs')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    l = plt.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc='lower right')
    l.set_zorder(300)

    session.close()

    if show:
        plt.show(block=True)
    else:
        filename = input('Filename?\t').lower()
        plt.savefig(filename)

    return plt


if __name__ == '__main__':
    #DB_PATH = 'postgresql+psycopg2:///generate_norwegian_flag_with_coord_input_2017-03-09T16:37:14.488425'
    #DB_PATH = 'postgresql+psycopg2:///majority_2017-03-19T18:55:08.922502'
    DB_PATH = 'postgresql+psycopg2:///majority_2017-03-24T15:48:10.659623'

    print(DB_PATH)
    db = get_db(DB_PATH)
    plot(db, show=True)
