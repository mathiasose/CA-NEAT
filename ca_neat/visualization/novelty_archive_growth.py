from collections import defaultdict
from neat.nn import create_feed_forward_phenotype
from sqlalchemy.sql.functions import func
from tqdm._tqdm import tqdm

from ca_neat.ca.analysis import serialize_cppn_rule
from ca_neat.database import get_db, Individual, Innovation
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.problems.novelty.generate_norwegian_flag_find_innovations import NEAT_CONFIG, CA_CONFIG

G = 1000

DB_PATH = 'postgresql+psycopg2:///generate_norwegian_flag_find_innovations_2017-05-26T18:38:44.84'


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

    scenarios = list(DB.get_scenarios(session=session))[:2]

    labels = {
        1: 'No speciation',
        2: 'With speciation',
        3: '-',
        4: '-',
        5: '-',
    }

    Xs = {}

    for scenario in scenarios:
        generation_population = session.query(Innovation.generation, func.count(Innovation.generation)) \
            .filter(Innovation.scenario_id == scenario.id) \
            .order_by(Innovation.generation) \
            .group_by(Innovation.generation) \
            .all()

        print(generation_population)

        s = 0
        Xs[scenario.id] = []

        for _, x in generation_population:
            s += x
            Xs[scenario.id].append(s)

    import matplotlib.pyplot as plt

    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind')

    fig, ax1 = plt.subplots()

    for scenario, color in zip(scenarios, palette):
        ax1.plot(Xs[scenario.id], color=color,
                linewidth=2,
                label=labels[scenario.id])

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Number of archived individuals')
    ax1.legend(loc='lower right')
    plt.title('Size of innovation archive')

    plt.ion()
    plt.show(block=True)
