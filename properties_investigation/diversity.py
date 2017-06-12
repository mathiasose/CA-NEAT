from collections import defaultdict
from neat.nn import create_feed_forward_phenotype
from tqdm._tqdm import tqdm

from ca_neat.ca.analysis import serialize_cppn_rule
from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.problems.novelty.generate_border_find_innovations import NEAT_CONFIG, CA_CONFIG

G = 1000

# DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-15T23:24:06.993649'
# DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-21T19:37:44.047610'
DB_PATH = 'postgresql+psycopg2:///generate_norwegian_flag_find_innovations_2017-05-26T18:38:44.84'

DB_PATH = 'postgresql+psycopg2:///generate_border_find_innovations_2017-04-21T20:14:55.212484'


# DB_PATH = 'postgresql+psycopg2:///synchronization_find_innovations_2017-05-30T21:46:52.833808'


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

    scenarios = list(DB.get_scenarios(session=session))

    labels = {
        1: 'No speciation',
        2: 'With speciation',
        3: '-',
        4: '-',
        5: '-',
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
                # n = len(s - {None})

                # ns[id].append(n)

                total_sets[id].update(s)
                total_development[id].append(len(total_sets[id]))

                pbar.update(1)

    session.close()

    for scenario in scenarios:
        print(scenario.id, total_development[scenario.id][-1])

    import matplotlib.pyplot as plt

    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind')

    # fig, ax1 = plt.subplots()
    #
    # for scenario, color in zip(scenarios, palette):
    #     ax1.plot(ns[scenario.id], color=color, linewidth=2, label=labels[scenario.id])
    #
    # ax1.set_xlabel('Generations')
    # ax1.set_ylabel('Number of behaviors')
    # ax1.legend(loc='upper right')
    # plt.title('Number of unique behaviors observed in each generation')

    fig, ax1 = plt.subplots()

    for scenario, color in zip(scenarios, palette):
        ax1.plot(total_development[scenario.id], color=color, linewidth=2, label=labels[scenario.id])

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Number of behaviors')
    ax1.legend(loc='upper left')
    plt.title('Cumulative number of unique behaviors observed')

    plt.ion()
    plt.show(block=True)
