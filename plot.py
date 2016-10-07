import matplotlib.pyplot as plt

from database import Db
from utils import pluck

INTERVAL = 5


def get_db(path):
    return Db(path, echo=False)


def plot(db_path, title, interval=None):
    db = get_db(db_path)
    session = db.Session()
    scenario = db.get_scenario(1, session=session)

    n_generations = scenario.generations
    last_generation = (n_generations - 1)
    generation_fitnesses = {}

    plt.ion()

    def draw():
        plt.clf()
        plt.axis([0, n_generations, -0.1, 1.1])
        plt.boxplot(tuple(generation_fitnesses.get(n, []) for n in range(n_generations)))
        plt.title(title)

    done = -1

    while True:
        for generation_n in range(done + 1, n_generations):
            generation = db.get_generation(scenario_id=1, generation=generation_n)

            if generation.count() < scenario.population_size:
                break
            else:
                generation_fitnesses[generation_n] = tuple(pluck(generation, 'fitness'))
                done = generation_n

        draw()

        if done == last_generation:
            break
        elif interval:
            plt.pause(interval)
        else:
            break

    plt.show(block=True)


if __name__ == '__main__':
    file = '2016-10-05 22:51:27.776011.db'
    plot('sqlite:///db/{}'.format(file), interval=INTERVAL)
