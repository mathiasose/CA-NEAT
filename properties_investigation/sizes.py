from statistics import mean, median

from collections import defaultdict
from tqdm._tqdm import tqdm

from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from properties_investigation.swiss_different_settings import NEAT_CONFIG

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
    means = defaultdict(list)
    medians = defaultdict(list)

    with tqdm(total=len(scenarios) * G) as pbar:
        for scenario in scenarios:
            for g in range(G):
                generation_population = DB.get_generation(scenario_id=scenario.id, generation=g, session=session) \
                    .with_entities(Individual.genotype)

                fs = [deserialize_gt(x.genotype, NEAT_CONFIG).size()[0] for x in generation_population]

                means[scenario.id].append(mean(fs))
                medians[scenario.id].append(median(fs))
                pbar.update(1)

    session.close()

    import matplotlib.pyplot as plt

    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind')

    plt.ion()

    for scenario, color in zip(scenarios, palette):
        plt.plot(means[scenario.id], color=color, linewidth=2, label=labels[scenario.id])

    plt.legend(loc='upper left')
    plt.gca().set_xlabel('Generations')
    plt.gca().set_ylabel('Nodes')
    plt.title('Mean number of nodes')
    plt.show(block=True)
