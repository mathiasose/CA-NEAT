import matplotlib.pyplot as plt
import seaborn
from matplotlib.colors import ListedColormap
from neat.config import Config
from neat.genes import ConnectionGene, NodeGene
from neat.genome import Genome
from neat.indexer import Indexer
from neat.nn import create_feed_forward_phenotype

config = Config()
config.input_nodes = 2
config.output_nodes = 1


def cppn_fig(cppn, r, c):
    plt.figure()
    f = lambda *args: cppn.serial_activate(args)[0]
    grid = tuple(
        tuple(
            f(x / c, y / c) for x in range(-r, r + 1)
        ) for y in range(-r, r + 1)
    )

    plt.imshow(
        grid,
        interpolation='none',
        extent=(-r, r, -r, r),
        cmap=ListedColormap(seaborn.color_palette('Blues'))
    )


if __name__ == '__main__':
    indexer = Indexer(0)
    i0 = NodeGene(indexer.get_next(), node_type='INPUT', activation_type='sigmoid')
    i1 = NodeGene(indexer.get_next(), node_type='INPUT', activation_type='sigmoid')
    h0 = NodeGene(indexer.get_next(), node_type='HIDDEN', activation_type='sin')
    h1 = NodeGene(indexer.get_next(), node_type='HIDDEN', activation_type='tanh')
    o0 = NodeGene(indexer.get_next(), node_type='OUTPUT', activation_type='identity')

    input_layer = (i0, i1)
    hidden_layer = (h0, h1)
    output_layer = (o0,)

    nodes = (input_layer + hidden_layer + output_layer)
    connections = (
        ConnectionGene(in_node_id=i0.ID, out_node_id=h0.ID, weight=0.5, enabled=True),
        ConnectionGene(in_node_id=i0.ID, out_node_id=h1.ID, weight=0.5, enabled=True),
        ConnectionGene(in_node_id=i1.ID, out_node_id=h0.ID, weight=0.5, enabled=True),
        ConnectionGene(in_node_id=i1.ID, out_node_id=h1.ID, weight=-0.5, enabled=True),
        ConnectionGene(in_node_id=h0.ID, out_node_id=o0.ID, weight=0.5, enabled=True),
        ConnectionGene(in_node_id=h1.ID, out_node_id=o0.ID, weight=0.5, enabled=True),
    )

    genome = Genome(indexer.get_next(), config, None, None)
    genome.node_genes = {n.ID: n for n in nodes}
    genome.conn_genes = {c.key: c for c in connections}

    seaborn.set(style='white')

    pt = create_feed_forward_phenotype(genome)

    cppn_fig(pt, r=10, c=10)
    cppn_fig(pt, r=20, c=10)
    cppn_fig(pt, r=20, c=20)
    plt.show()
