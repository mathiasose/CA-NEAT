import os
from statistics import mean, median

import matplotlib.pyplot as plt
import seaborn
from sqlalchemy.sql.functions import func
from tqdm._tqdm import tqdm

from ca_neat.database import Db, Individual, Scenario, get_db
from ca_neat.utils import PROJECT_ROOT

seaborn.set_style('white')


def plot(db: Db, show=True, generations=None):
    session = db.Session()

    Xs = session.query(Individual) \
        .filter(Individual.fitness >= 1.0) \
        .order_by(Individual.scenario_id, Individual.generation) \
        .distinct(Individual.scenario_id)

    gens = [i.generation for i in Xs]

    print(sorted(gens))
    print(len(gens))

    scenarios = list(db.get_scenarios(session=session))
    generations = generations or scenarios[0].generations or max(gens)

    assert scenarios

    scenarios = list(filter(lambda s: db.get_individuals(s.id, session).count(), scenarios))

    scenarios_fitnesses = {scenario.id: [] for scenario in scenarios}
    for scenario in tqdm(scenarios):
        for g in range(generations):
            max_f = session.query(Individual, func.max(Individual.fitness)) \
                .filter(Individual.scenario_id == scenario.id, Individual.generation == g) \
                .value(func.max(Individual.fitness))

            if not max_f:
                break

            scenarios_fitnesses[scenario.id].append(max_f)

    for k, sf in scenarios_fitnesses.items():
        while len(sf) < generations:
            sf.append(sf[-1])

    if show:
        plt.ion()

    plt.title(scenarios[0].description.split('\n')[0])
    plt.xlabel('Generations')

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.set_zorder(200)
    ax2.set_zorder(100)
    ax1.patch.set_visible(False)

    ax1.set_color_cycle(seaborn.color_palette('hls', n_colors=len(scenarios)))
    ax1.axis([0, generations, 0.0, 1.1])
    ax1.set_ylabel('Fitness')

    ax2.set_ylabel('Finished runs')
    ax2.set_ylim([0, int(len(scenarios) * 1.1)])

    means = [mean([scenarios_fitnesses[s.id][g] for s in scenarios]) for g in range(generations)]
    medians = [median([scenarios_fitnesses[s.id][g] for s in scenarios]) for g in range(generations)]

    ax1.plot(means, 'r', label='mean', zorder=210)
    ax1.plot(medians, 'k--', label='median', zorder=211)

    ax2.hist(gens, bins=generations, range=(0, generations), cumulative=True, color=['pink'], zorder=110,
             label='Finished runs', histtype='stepfilled')
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
    #DB_PATH = 'postgresql+psycopg2:///generate_border_with_coord_input_2017-05-25T14:11:14.401192'
    #DB_PATH = 'postgresql+psycopg2:///synchronization_2017-05-06T17:00:23.012278'
    DB_PATH = 'postgresql+psycopg2:///generate_norwegian_flag_with_coord_input_2017-03-09T16:37:14.48'
    # DB_PATH = 'postgresql+psycopg2:///generate_norwegian_flag_with_coord_input_2017-05-25T15:20:52.64'

    print(DB_PATH)
    db = get_db(DB_PATH)
    plot(db, show=True)
