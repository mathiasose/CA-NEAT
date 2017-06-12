import os
from tkinter import TclError

import matplotlib.pyplot as plt

from ca_neat.database import Db, get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.problems.majority import NEAT_CONFIG
from ca_neat.utils import PROJECT_ROOT, pluck


def plot_fitnesses_over_generations(db: Db, scenario_id: int, title=None, action='show', step=1, generations=None):
    session = db.Session()
    scenario = db.get_scenario(scenario_id, session=session)

    assert scenario

    generation_fitnesses = {}
    n_species = {}

    if action == 'show':
        plt.ion()
    else:
        plt.ioff()

    def draw():
        plt.clf()
        fig = plt.gcf()
        ax1 = plt.gca()

        plt.title(title or scenario.description)
        # fig.canvas.set_window_title(os.path.basename(DB_PATH))

        last_gen = max(generation_fitnesses.keys()) + 1
        ax1.axis([0, last_gen, -0.1, 1.1])

        n = generation
        ax1.boxplot(
            x=tuple(generation_fitnesses[n] for n in sorted(generation_fitnesses.keys())),
            # positions=range(0, len(generation_fitnesses)),
            labels=sorted(generation_fitnesses.keys()),
        )
        ax1.set_ylabel('fitness')
        ax1.set_xlabel('generation')

        # ax2 = ax1.twinx()
        # ax2.axis([0, last_gen, 0, max(n_species.values()) + 1])
        # ax2.plot(
        #     range(0, last_gen // step),
        #     tuple(n_species.get(n, 0) for n in range(0, last_gen)),
        #     'go'
        # )
        # ax2.set_ylabel('number of species')

        return fig

    generations = generations or scenario.generations
    for generation_n in range(0, generations, step):
        generation = db.get_generation(scenario_id=scenario_id, generation=generation_n, session=session)
        if generation.count() == 0:
            break

        n_species[generation_n] = generation \
            .order_by(Individual.species) \
            .distinct(Individual.species) \
            .count()

        print('Generation {}: {} individuals / {} species'.format(
            generation_n,
            generation.count(),
            n_species[generation_n])
        )
        generation_fitnesses[generation_n] = tuple(pluck(generation, 'fitness'))

    draw()

    if action == 'show':
        plt.show(block=True)

    return plt


if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'problems', 'results'))
    DB_PATH = 'postgresql+psycopg2:///majority_2017-03-19T18:55:08.922502'

    print(DB_PATH)
    db = get_db(DB_PATH)
    for scenario in db.get_scenarios():
        scenario_id = scenario.id
        try:
            plot_fitnesses_over_generations(db, scenario_id=scenario_id, step=5, generations=100)
        except TclError:
            pass
