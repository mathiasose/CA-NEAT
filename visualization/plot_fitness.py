import os

import matplotlib.pyplot as plt

from database import Db
from utils import pluck


def get_db(path):
    return Db(path, echo=False)


def plot_fitnesses_over_generations(db_path, title=None, interval=None):
    db = get_db(db_path)
    session = db.Session()
    scenario = db.get_scenario(1, session=session)

    assert scenario

    n_generations = scenario.generations
    last_generation = (n_generations - 1)

    plt.ion()
    generation_fitnesses = {}
    n_species = {}

    def draw():
        plt.clf()
        fig = plt.gcf()
        ax1 = plt.gca()

        plt.title(title or scenario.description)
        fig.canvas.set_window_title(os.path.basename(db_path))

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

    done = -1

    while True:
        for generation_n in range(done + 1, n_generations):
            generation = db.get_generation(scenario_id=1, generation=generation_n)
            genotypes = pluck(generation, 'genotype')
            n_species[generation_n] = len(set(gt.species_id for gt in genotypes))

            if generation.count() < scenario.population_size:
                print('Generation {} at {}/{} individuals completed'.format(
                    generation_n,
                    generation.count(),
                    scenario.population_size,
                ))
                break
            else:
                print('Generation {}: {} species'.format(generation_n, n_species[generation_n]))
                generation_fitnesses[generation_n] = tuple(pluck(generation, 'fitness'))
                done = generation_n

        draw()

        if done == last_generation:
            print('Finish after generation {}'.format(last_generation))
            break

        elif interval:
            plt.pause(interval)
        else:
            break

    plt.show(block=True)


if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    RESULTS_DIR = os.path.abspath(os.path.join(THIS_FILE, '..', '..', 'problems', 'results'))
    file = os.path.join(RESULTS_DIR, 'replicate_twocolor/', '2016-11-07 23:33:03.662880.db')
    db_path = 'sqlite:///{}'.format(file)
    print(db_path)
    plot_fitnesses_over_generations(db_path, interval=300)
