from collections import defaultdict
from statistics import mean

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
        5: 'A',
    }
    means = {}

    # bins = [
    #     (0.0, 0.0),
    #     (0.0, 0.20),
    #     (0.20, 0.40),
    #     (0.40, 0.60),
    #     (0.60, 0.80),
    #     (0.80, 1.0),
    #     (1.0, 1.0),
    # ]
    bins = [
        (0.0, 0.0),
        (0.0, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.0),
        (1.0, 1.0),
    ]
    buckets = {}

    with tqdm(total=len(scenarios) * G) as pbar:
        for scenario in scenarios:
            buckets[scenario.id] = {}
            means[scenario.id] = []
            for g in range(G):
                buckets[scenario.id][g] = {i: 0 for i in range(len(bins))}
                generation_population = DB.get_generation(scenario_id=scenario.id, generation=g, session=session) \
                    .with_entities(Individual.λ)

                λs = [x.λ for x in generation_population]
                for λ in λs:
                    for i, (lo, hi) in enumerate(bins):
                        if (hi == lo == λ) or (lo <= λ < hi):
                            buckets[scenario.id][g][i] += 1
                            break

                s = sum(buckets[scenario.id][g].values())

                assert s == len(λs)

                for k, v in buckets[scenario.id][g].items():
                    buckets[scenario.id][g][k] = v / s

                means[scenario.id].append(mean(λs))
                pbar.update(1)

    session.close()

    import matplotlib.pyplot as plt

    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind', n_colors=len(bins))

    fig1 = plt.figure()

    for scenario in scenarios:
        plt.plot(means[scenario.id], label=labels[scenario.id])

    plt.gca().set_ylabel('λ')
    plt.gca().set_xlabel('Generations')
    plt.legend(loc='lower left')
    plt.title('Mean λ of five diverging populations')

    for scenario in scenarios:
        fig2 = plt.figure()

        lambdalabels = [
            f'λ = {a}' if a == b else f'{a} < λ <= {b}' for a, b in bins
            ]
        lambdalabels[-2] = '{} < λ < {}'.format(*bins[-2])
        for color, label in reversed(list(zip(palette, lambdalabels))):
            plt.plot([], [], color=color, label=label, linewidth=5)

        l = [[buckets[scenario.id][g][i] for g in range(G)] for i in range(len(bins))]
        plt.ylim(0.0, 1.0)
        plt.stackplot(range(G), *l, colors=palette)
        plt.gca().set_xlabel('Generations')
        plt.gca().set_ylabel('Distribution')
        plt.title(f'λ distribution in {labels[scenario.id]}')
        plt.legend(loc='upper center', frameon=True)

    plt.ion()
    plt.show(block=True)
