from statistics import mean

from neat.nn import create_feed_forward_phenotype
from tqdm._tqdm import tqdm

from ca_neat.ca.analysis import calculate_sensitivity, calculate_dominance
from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from properties_investigation.swiss_different_settings import CA_CONFIG, NEAT_CONFIG

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
        5: 'A',
    }
    mean_λs = {}
    mean_sensitivities = {}
    mean_dominances = {}

    with tqdm(total=len(scenarios) * G) as pbar:
        for scenario in scenarios:
            mean_λs[scenario.id] = []
            mean_sensitivities[scenario.id] = []
            mean_dominances[scenario.id] = []
            for g in range(G):
                generation_population = DB.get_generation(scenario_id=scenario.id, generation=g, session=session) \
                    .with_entities(Individual.λ, Individual.genotype)

                mean_λs[scenario.id].append(mean(x.λ for x in generation_population))

                sensitivities = []
                dominances = []
                for ind in generation_population:
                    gt = deserialize_gt(ind.genotype, NEAT_CONFIG)
                    pt = create_feed_forward_phenotype(gt)
                    sensitivities.append(calculate_sensitivity(pt, CA_CONFIG))
                    dominances.append(calculate_dominance(pt, CA_CONFIG))

                mean_sensitivities[scenario.id].append(mean(sensitivities))
                mean_dominances[scenario.id].append(mean(dominances))
                pbar.update(1)

    session.close()

    import matplotlib.pyplot as plt

    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind', n_colors=len(scenarios))

    fig1 = plt.figure()

    for scenario in scenarios:
        plt.plot(mean_λs[scenario.id], label=labels[scenario.id])

    plt.gca().set_ylabel('λ')
    plt.gca().set_xlabel('Generations')
    plt.legend(loc='lower left')
    plt.title('Mean λ of five diverging populations')

    fig2 = plt.figure()

    for scenario in scenarios:
        plt.plot(mean_sensitivities[scenario.id], label=labels[scenario.id])

    plt.gca().set_ylabel('Sensitivity')
    plt.gca().set_xlabel('Generations')
    plt.legend(loc='lower left')
    plt.title('Mean sensitivity of five diverging populations')

    fig3 = plt.figure()

    for scenario in scenarios:
        plt.plot(mean_dominances[scenario.id], label=labels[scenario.id])

    plt.gca().set_ylabel('Dominance')
    plt.gca().set_xlabel('Generations')
    plt.legend(loc='lower left')
    plt.title('Mean dominance of five diverging populations')

    plt.ion()
    plt.show(block=True)
