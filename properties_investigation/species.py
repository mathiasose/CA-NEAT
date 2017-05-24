from statistics import mean

from collections import defaultdict, Counter
from neat.nn import create_feed_forward_phenotype
from tqdm._tqdm import tqdm

from ca_neat.ca.analysis import serialize_cppn_rule
from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from properties_investigation.swiss_different_settings import NEAT_CONFIG, CA_CONFIG

G = 100

#DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-15T23:24:06.993649'
DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-21T19:37:44.047610'

if __name__ == '__main__':
    DB = get_db(DB_PATH)
    session = DB.Session()

    scenarios = list(DB.get_scenarios(session=session))[::-1][3:]

    labels = {
        1: 'E',
        2: 'D',
        3: 'C',
        4: 'B',
        5: 'A',
    }

    ns = {k: [] for k in labels.keys()}
    sizes = {k: [] for k in labels.keys()}
    maxs = {k: [] for k in labels.keys()}

    with tqdm(total=len(scenarios) * G) as pbar:
        for scenario in scenarios:
            for g in range(G):
                id = scenario.id
                generation_population = DB.get_generation(scenario_id=id, generation=g, session=session) \
                    .with_entities(Individual.species)

                enums = [individual.species for individual in generation_population]

                s = set(enums)
                n = len(s)

                ns[id].append(n)

                c = Counter(enums)
                sizes[id].append(mean(c.values()))
                maxs[id].append(max(c.values()))

                pbar.update(1)

    session.close()

    import matplotlib.pyplot as plt

    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind')[3:]
    colors = {s.id: c for s, c in zip(scenarios, palette)}

    fig, ax1 = plt.subplots()

    for scenario, color in zip(scenarios, palette):
        ax1.plot(ns[scenario.id], color=color, linewidth=2, label=labels[scenario.id])

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Number of species')
    ax1.legend(loc='upper left')
    plt.title('Number of species in each generation')

    fig, ax1 = plt.subplots()

    for scenario in scenarios:
        ax1.plot(sizes[scenario.id], color=colors[scenario.id], linewidth=2,
                 label='{} mean'.format(labels[scenario.id]))
        ax1.plot(maxs[scenario.id], color=colors[scenario.id], linewidth=2, linestyle='dotted',
                 label='{} max'.format(labels[scenario.id]))

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Size')
    ax1.legend(loc='upper right')
    plt.title('Number of individuals per species')

    plt.ion()
    plt.show(block=True)
