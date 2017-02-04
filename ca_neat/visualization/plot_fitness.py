import os
from tkinter import TclError

import matplotlib.pyplot as plt

from ca_neat.database import Db, get_db
from ca_neat.utils import PROJECT_ROOT, pluck


def plot_fitnesses_over_generations(db: Db, scenario_id: int, title=None, interval=None, action='show'):
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
        # fig.canvas.set_window_title(os.path.basename(db_path))

        last_gen = max(generation_fitnesses.keys()) + 1
        ax1.axis([0, last_gen, -0.1, 1.1])
        ax1.boxplot(
            x=tuple(generation_fitnesses.get(n, []) for n in range(0, last_gen)),
            positions=range(0, last_gen),
            labels=tuple(i if i % 10 == 0 else '' for i in range(0, last_gen)),
        )
        ax1.set_ylabel('fitness')
        ax1.set_xlabel('generation')

        ax2 = ax1.twinx()
        ax2.axis([0, last_gen, 0, max(n_species.values()) + 1])
        ax2.plot(
            range(0, last_gen),
            tuple(n_species.get(n, 0) for n in range(0, last_gen)),
            'go'
        )
        ax2.set_ylabel('number of species')

        return fig

    done = 0
    while True:
        for generation_n in range(done, scenario.generations):
            generation = db.get_generation(scenario_id=scenario_id, generation=generation_n, session=session)
            if generation.count() == 0:
                done = generation_n - 2
                break

            genotypes = pluck(generation, 'genotype')
            n_species[generation_n] = len(set(gt.species_id for gt in genotypes))

            print('Generation {}: {} individuals / {} species'.format(
                generation_n,
                generation.count(),
                n_species[generation_n])
            )
            generation_fitnesses[generation_n] = tuple(pluck(generation, 'fitness'))

        draw()

        if action == 'show' and interval:
            plt.pause(interval)
        else:
            break

    if action == 'show':
        plt.show(block=True)

    return plt


if __name__ == '__main__':
    problem_name = 'generate_border'
    db_file = '2016-11-23 13:51:24.771958.db'
    scenario_id = 47

    THIS_FILE = os.path.abspath(__file__)
    RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'problems', 'results'))
    file = os.path.join(RESULTS_DIR, problem_name, db_file)
    db_path = 'sqlite:///{}'.format(file)

    print(db_path)
    db = get_db(db_path)
    for scenario in [scenario_id]:
        try:
            plot_fitnesses_over_generations(db, scenario_id=scenario_id, interval=300)
        except TclError:
            pass
