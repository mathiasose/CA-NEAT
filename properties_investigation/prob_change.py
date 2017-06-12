from statistics import median, mean

from collections import defaultdict
from distance._simpledists import hamming
from neat.nn import create_feed_forward_phenotype
from tqdm._tqdm import tqdm

from ca_neat.ca.analysis import serialize_cppn_rule
from ca_neat.database import get_db, Individual
from ca_neat.ga.population import create_initial_population
from ca_neat.ga.serialize import deserialize_gt
from properties_investigation.swiss_different_settings import NEAT_CONFIG, CA_CONFIG

# DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-15T23:24:06.993649'
from properties_investigation.vestigial import find_nodes_connected_to_output

DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-21T19:37:44.047610'
G = 100
SCENARIO = 5


def enumerate_safe(gt):
    try:
        _, outputs = serialize_cppn_rule(
            create_feed_forward_phenotype(
                gt
            ), CA_CONFIG
        )
        return outputs
    except OverflowError:
        return None


if __name__ == '__main__':
    DB = get_db(DB_PATH)

    prevs = {}
    changes = defaultdict(list)
    sizes = defaultdict(list)
    l = defaultdict(list)

    individuals = DB.get_individuals(scenario_id=SCENARIO) \
        .filter(Individual.generation < G) \
        .order_by(Individual.generation, Individual.individual_number) \
        .with_entities(Individual.genotype, Individual.individual_number, Individual.generation)

    for individual in tqdm(individuals, total=G * 1000):
        gt = deserialize_gt(individual.genotype, NEAT_CONFIG)
        enum = enumerate_safe(gt)
        num_hidden, _ = gt.size()

        reachable = len(find_nodes_connected_to_output(gt))
        l[individual.generation].append((gt.num_inputs + gt.num_outputs + num_hidden) - reachable)

        prev = prevs.get(individual.individual_number)
        if prev:
            changes[individual.generation].append(prev != enum)
        sizes[individual.generation].append(num_hidden)

        # if prev:
        #     distance = hamming(prev, enum, normalized=True)
        #     ds[individual.generation].append(distance)

        prevs[individual.individual_number] = enum

    distances = []
    means = []
    medians = []
    mean_changes = []
    mean_size = []
    median_size = []
    unreachables = []

    for g in range(G):
        # means.append(mean(l))
        # medians.append(median(l))
        # distances.extend(l)
        c = changes.get(g)
        if c:
            mean_changes.append(mean(c))
        mean_size.append(mean(sizes[g]))
        median_size.append(median(sizes[g]))
        unreachables.append(mean(l[g]))

    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    lns1 = ax1.plot(mean_changes, label='Ratio of behaviors changed', linewidth=2, color=palette[0])
    lns2 = ax2.plot(mean_size, label='Mean number of hidden nodes', linewidth=1, color=palette[1])
    lns3 = ax2.plot(unreachables, label='Mean number of unreachable nodes', linestyle='dotted', color=palette[2])

    ax2.grid(None)

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Ratio')
    ax2.set_ylabel('Number of nodes')

    plt.ion()
    plt.title('Change in k after mutation, correlated with number of hidden nodes and number of unreachable nodes')
    plt.show(block=True)
