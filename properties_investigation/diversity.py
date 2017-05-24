from collections import defaultdict
from neat.nn import create_feed_forward_phenotype
from tqdm._tqdm import tqdm

from ca_neat.ca.analysis import serialize_cppn_rule
from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from properties_investigation.swiss_different_settings import NEAT_CONFIG, CA_CONFIG

G = 100

#DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-15T23:24:06.993649'
DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-21T19:37:44.047610'


def enumerate_to_str_safe(ind):
    try:
        _, outputs = serialize_cppn_rule(
            create_feed_forward_phenotype(
                deserialize_gt(ind.genotype, NEAT_CONFIG)
            ), CA_CONFIG
        )
        return ''.join(outputs)
    except OverflowError:
        return None


if __name__ == '__main__':
    DB = get_db(DB_PATH)
    session = DB.Session()

    scenarios = list(DB.get_scenarios(session=session))[::-1]

    labels = {
        1: 'E',
        2: 'D',
        3: 'C',
        4: 'B',
        5: 'A',
    }
    ns = {k: [] for k in labels.keys()}
    total_sets = {k: set() for k in labels.keys()}
    total_development = {k: [] for k in labels.keys()}

    with tqdm(total=len(scenarios) * G) as pbar:
        for scenario in scenarios:
            for g in range(G):
                id = scenario.id
                generation_population = DB.get_generation(scenario_id=id, generation=g, session=session) \
                    .with_entities(Individual.genotype)

                enums = [enumerate_to_str_safe(individual) for individual in generation_population]

                s = set(enums)
                n = len(s - {None})

                ns[id].append(n)

                total_sets[id].update(s)
                total_development[id].append(len(total_sets[id]))

                pbar.update(1)

    session.close()

    import matplotlib.pyplot as plt

    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind')

    fig, ax1 = plt.subplots()

    for scenario, color in zip(scenarios, palette):
        ax1.plot(ns[scenario.id], color=color, linewidth=2, label=labels[scenario.id])

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Number of behaviors')
    ax1.legend(loc='upper right')
    plt.title('Number of unique behaviors observed in each generation')

    fig, ax1 = plt.subplots()

    for scenario, color in zip(scenarios, palette):
        ax1.plot(total_development[scenario.id], color=color, linewidth=2, label=labels[scenario.id])

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Number of behaviors')
    ax1.legend(loc='upper left')
    plt.title('Cummulative number of unique behaviors observed')

    plt.ion()
    plt.show(block=True)
