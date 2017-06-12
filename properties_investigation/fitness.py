from statistics import mean

from collections import defaultdict
from tqdm._tqdm import tqdm

from ca_neat.database import get_db, Individual

G = 100

#DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-15T23:24:06.993649'
DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-21T19:37:44.047610'

if __name__ == '__main__':
    DB = get_db(DB_PATH)
    session = DB.Session()

    scenarios = list(DB.get_scenarios(session=session))[::-1]

    labels = {
        1: 'E',
        2: 'D',
        3: 'C',
        4: 'B',
        5: 'T',
    }
    maxes = defaultdict(list)
    means = defaultdict(list)

    with tqdm(total=len(scenarios) * G) as pbar:
        for scenario in scenarios:
            for g in range(G):
                generation_population = DB.get_generation(scenario_id=scenario.id, generation=g, session=session) \
                    .with_entities(Individual.fitness)

                fs = [x.fitness for x in generation_population]

                if not fs:
                    continue

                maxes[scenario.id].append(max(fs))
                means[scenario.id].append(mean(fs))
                pbar.update(1)

    session.close()

    import matplotlib.pyplot as plt

    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind')

    plt.figure()
    for scenario, color in zip(scenarios, palette):
        plt.plot(means[scenario.id], color=color, linewidth=2, label=labels[scenario.id])

    plt.legend(loc='upper left')
    plt.gca().set_xlabel('Generations')
    plt.gca().set_ylabel('Fitness')
    plt.title('Mean fitness of five diverging populations')

    fig, axs = plt.subplots(nrows=len(scenarios), sharex=True, sharey=True)
    plt.suptitle('Max fitness of five diverging populations')
    for ax, scenario, color in zip(axs, scenarios, palette):
        ax.plot(maxes[scenario.id], color=color, label='{} max fitness'.format(labels[scenario.id]))
        ax.set_ylabel('Fitness')
        ax.set_title(labels[scenario.id])
    ax.set_xlabel('Generations')

    # plt.legend(loc='upper left')

    plt.ion()
    plt.show(block=True)
