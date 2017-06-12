import os
from datetime import datetime

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import LLLCRRR
from ca_neat.patterns.patterns import ALPHABET_2
from ca_neat.run_novelty_experiment import initialize_scenario

P = 200
E = 0
G = 1
M = 1

CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_2
CA_CONFIG.neighbourhood = LLLCRRR
CA_CONFIG.iterations = M
CA_CONFIG.compute_lambda = False

NEAT_CONFIG = CPPNNEATConfig()
NEAT_CONFIG.do_speciation = False
NEAT_CONFIG.innovation_threshold = 0.5
NEAT_CONFIG.survival_threshold = 0.5

NEAT_CONFIG.pop_size = P
NEAT_CONFIG.generations = G
NEAT_CONFIG.elitism = E

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

FITNESS_F = None
PAIR_SELECTION_F = sigma_scaled

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')

    dt = datetime.now().isoformat()
    DB_PATH = 'postgresql+psycopg2:///synchronization_find_innovations_2017-05-30T21:46:52.833808'

    DESCRIPTION = '"Synchronization problem"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    NEAT_CONFIG.do_speciation = False
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=PAIR_SELECTION_F,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )
    #
    # NEAT_CONFIG.do_speciation = True
    # initialize_scenario(
    #     db_path=DB_PATH,
    #     description=DESCRIPTION,
    #     fitness_f=FITNESS_F,
    #     pair_selection_f=PAIR_SELECTION_F,
    #     neat_config=NEAT_CONFIG,
    #     ca_config=CA_CONFIG,
    # )
