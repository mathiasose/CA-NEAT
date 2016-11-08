from neat.config import Config


class CPPNNEATConfig(Config):
    generations = None
    stagnation_limit = None
    survival_threshold = None
    elitism = 0
    initial_hidden_nodes = 0

    def __init__(self, **kwargs):
        from neat import activation_functions
        from neat.genes import NodeGene, ConnectionGene
        from neat.genome import Genome

        super().__init__()

        self.output_nodes = 1
        self.genotype = Genome
        self.node_gene_type = NodeGene
        self.conn_gene_type = ConnectionGene
        self.activation_functions = tuple(activation_functions.functions.keys())  # all the possible functions


class CAConfig:
    neighbourhood = None
    alphabet = None
    geometry = None
    iterations = None
    r = None
    initial = None
    etc = {}
