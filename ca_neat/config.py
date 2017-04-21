import random
from typing import Any, Dict

from neat.config import Config


class CPPNNEATConfig(Config):
    generations = None
    elitism = 0
    initial_hidden_nodes = 0
    stagnation_median_threshold = True
    stop_when_optimal_found = True
    initial_connection = 'partial'
    connection_fraction = lambda *args: 0.5 * (1 + random.random())

    do_speciation = True
    survival_threshold = 0.2
    stagnation_limit = 15

    compatibility_threshold = 3.0
    excess_coefficient = 1.0
    disjoint_coefficient = 1.0
    weight_coefficient = 0.5

    max_weight = 30
    min_weight = -30
    weight_stdev = 1.0

    prob_add_conn = 0.5
    prob_add_node = 0.5
    prob_delete_conn = 0.25
    prob_delete_node = 0.25
    prob_mutate_bias = 0.8
    bias_mutation_power = 0.5
    prob_mutate_response = 0.8
    response_mutation_power = 0.5
    prob_mutate_weight = 0.8
    prob_replace_weight = 0.1
    weight_mutation_power = 0.5
    prob_mutate_activation = 0.002
    prob_toggle_link = 0.01

    def __init__(self, **kwargs):
    innovation_threshold = 3.0
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
    initial = None
    etc: Dict[str, Any] = {}
    compute_lambda = False
