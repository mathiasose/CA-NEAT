import os
from datetime import datetime

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import VON_NEUMANN
from ca_neat.patterns.patterns import ALPHABET_4, SEED_6X6, TRICOLOR
from ca_neat.problems.common import morphogenesis_fitness_f
from ca_neat.run_experiment import initialize_scenario

CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_4
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 30
CA_CONFIG.etc = {
    'target_pattern': TRICOLOR,
    'seed': SEED_6X6,
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 10000
NEAT_CONFIG.elitism = 1
NEAT_CONFIG.survival_threshold = 0.2
NEAT_CONFIG.stagnation_limit = 15

NEAT_CONFIG.compatibility_threshold = 3.0
NEAT_CONFIG.excess_coefficient = 1.0
NEAT_CONFIG.disjoint_coefficient = 1.0
NEAT_CONFIG.weight_coefficient = 0.5

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

NEAT_CONFIG.max_weight = 30
NEAT_CONFIG.min_weight = -30
NEAT_CONFIG.weight_stdev = 1.0

NEAT_CONFIG.prob_add_conn = 0.5
NEAT_CONFIG.prob_add_node = 0.5
NEAT_CONFIG.prob_delete_conn = 0.25
NEAT_CONFIG.prob_delete_node = 0.25
NEAT_CONFIG.prob_mutate_bias = 0.8
NEAT_CONFIG.bias_mutation_power = 0.5
NEAT_CONFIG.prob_mutate_response = 0.8
NEAT_CONFIG.response_mutation_power = 0.5
NEAT_CONFIG.prob_mutate_weight = 0.8
NEAT_CONFIG.prob_replace_weight = 0.1
NEAT_CONFIG.weight_mutation_power = 0.5
NEAT_CONFIG.prob_mutate_activation = 0.002
NEAT_CONFIG.prob_toggle_link = 0.01

PAIR_SELECTION_F = sigma_scaled
FITNESS_F = morphogenesis_fitness_f

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'results', PROBLEM_NAME))
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    DB_PATH = 'sqlite:///' + os.path.join(RESULTS_DIR, '{}.DB'.format(datetime.now()))

    DESCRIPTION = '"Tricolor flag morphogenesis"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(100):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
