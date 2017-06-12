from statistics import mean

from collections import defaultdict
from neat.genome import Genome
from tqdm._tqdm import tqdm

from ca_neat.database import get_db
from ca_neat.ga.serialize import deserialize_gt
from properties_investigation.swiss_different_settings import NEAT_CONFIG

# DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-15T23:24:06.993649'
DB_PATH = 'postgresql+psycopg2:///swiss_different_settings_2017-05-21T19:37:44.047610'
G = 100


def find_nodes_connected_to_output(genome: Genome):
    outputs = set(ng for ng in genome.node_genes.values() if ng.type == 'OUTPUT')
    connections = [c for c in genome.conn_genes.values() if c.enabled]

    can_reach_from_output = set(ng.ID for ng in outputs)

    finished = False
    while not finished:
        finished = True
        for connection in connections:
            a = connection.in_node_id
            b = connection.out_node_id

            # if k not in can_reach_from_input and a in can_reach_from_input:
            #     can_reach_from_input.add(k)
            #     finished = False

            if a not in can_reach_from_output and b in can_reach_from_output:
                can_reach_from_output.add(a)
                finished = False

    return can_reach_from_output


if __name__ == '__main__':
    DB = get_db(DB_PATH)
    session = DB.Session()
    scenarios = list(DB.get_scenarios(session=session))[::-1]

    means = defaultdict(list)
    rs = defaultdict(list)
    with tqdm(total=len(scenarios) * G) as pbar:
        for scenario in scenarios:
            for g in range(G):
                l = []
                t = []
                for individual in DB.get_generation(scenario_id=scenario.id, generation=g, session=session):
                    gt = deserialize_gt(individual.genotype, NEAT_CONFIG)
                    num_hidden, _ = gt.size()
                    reachable = len(find_nodes_connected_to_output(gt))
                    tot = gt.num_inputs + gt.num_outputs + num_hidden
                    t.append(reachable / tot)
                    l.append(tot - reachable)

                rs[scenario.id].append(mean(t))
                means[scenario.id].append(mean(l))

                pbar.update(1)

    session.close()

    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_style('whitegrid')
    palette = seaborn.palettes.color_palette('colorblind')

    labels = {
        1: 'E',
        2: 'D',
        3: 'C',
        4: 'B',
        5: 'T',
    }

    fig = plt.figure()

    for scenario in scenarios:
        plt.plot(means[scenario.id], label=labels[scenario.id])

    plt.legend(loc='upper left')
    plt.gca().set_xlabel('Generations')
    plt.gca().set_ylabel('Number of disconnected nodes')
    plt.title('Mean number of disconnected nodes')

    fig = plt.figure()

    for scenario in scenarios:
        plt.plot(rs[scenario.id], label=labels[scenario.id])

    plt.legend(loc='lower left')
    plt.gca().set_xlabel('Generations')
    plt.gca().set_ylabel('Ratio of connected nodes')
    plt.title('Mean ratio of connected nodes')

    plt.ion()
    plt.show(block=True)
