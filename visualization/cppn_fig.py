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
    f = lambda *args: pt.serial_activate(args)[0]
    R = 10
    c = 1
    grid = tuple(
        tuple(
            f(x / c, y / c) for x in range(-R * c, R * c)
        ) for y in range(-R * c, R * c)
    )
    print(*grid, sep='\n')
    print(min(min(row) for row in grid))
    print(max(max(row) for row in grid))

    plt.imshow(
        grid,
        interpolation='nearest',
        extent=(-R, R, -R, R),
        cmap=ListedColormap(seaborn.color_palette('hls'))
    )
    plt.show()
