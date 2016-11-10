import os

import matplotlib.pyplot as plt

from database import Db
from utils import pluck, PROJECT_ROOT


def get_db(path):
    return Db(path, echo=False)


def plot_fitnesses_over_generations(db: Db, scenario_id: int, title=None, interval=None, action='show'):
    session = db.Session()
    scenario = db.get_scenario(scenario_id, session=session)

    assert scenario

    n_generations = scenario.generations

    plt.ion()
    generation_fitnesses = {}
    n_species = {}

    def draw():
        plt.clf()
        fig = plt.gcf()
        ax1 = plt.gca()

        plt.title(title or scenario.description)
        # fig.canvas.set_window_title(os.path.basename(db_path))

        ax1.axis([0, n_generations, -0.1, 1.1])
        ax1.boxplot(
            x=tuple(generation_fitnesses.get(n, []) for n in range(0, n_generations)),
            positions=range(0, n_generations),
        )
        ax1.set_ylabel('fitness')
        ax1.set_xlabel('generation')

        ax2 = ax1.twinx()
        ax2.axis([0, n_generations, 0, max(n_species.values()) + 1])
        ax2.plot(
            range(0, n_generations),
            tuple(n_species.get(n, 0) for n in range(0, n_generations)),
            'go'
        )
        ax2.set_ylabel('number of species')

        return fig

    while True:
        for generation_n in range(n_generations):
            generation = db.get_generation(scenario_id=scenario_id, generation=generation_n)
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
    problem_dir = 'replicate_tricolor/'
    db_file = '2016-11-09 20:17:10.035768.db'
    scenario_id = 3

    THIS_FILE = os.path.abspath(__file__)
    RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'problems', 'results'))
    file = os.path.join(RESULTS_DIR, problem_dir, db_file)
    db_path = 'sqlite:///{}'.format(file)

    print(db_path)
    plot_fitnesses_over_generations(get_db(db_path), scenario_id=scenario_id, interval=300)
