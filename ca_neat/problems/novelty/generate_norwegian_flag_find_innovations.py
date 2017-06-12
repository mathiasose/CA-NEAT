import os
from datetime import datetime

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import VON_NEUMANN
from ca_neat.patterns.patterns import ALPHABET_2, BORDER, SEED_6X6, ALPHABET_4, NORWEGIAN, SEED_7X7
from ca_neat.problems.common import morphogenesis_fitness_f
from ca_neat.run_novelty_experiment import initialize_scenario

CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_4
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 30
CA_CONFIG.etc = {
    'target_pattern': NORWEGIAN,
    'seed': SEED_7X7,
}

NEAT_CONFIG = CPPNNEATConfig()
NEAT_CONFIG.do_speciation = False
NEAT_CONFIG.innovation_threshold = 0.5

NEAT_CONFIG.pop_size = 50
NEAT_CONFIG.generations = 1000
NEAT_CONFIG.elitism = 0

NEAT_CONFIG.survival_threshold = 0.5

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)

PAIR_SELECTION_F = sigma_scaled
FITNESS_F = morphogenesis_fitness_f

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')

    # DB_PATH = 'postgresql+psycopg2:///generate_norwegian_flag_find_innovations_2017-05-26T17:21:47.724134'
    DB_PATH = 'postgresql+psycopg2:///{}_{}'.format(PROBLEM_NAME, datetime.now().isoformat())

    DESCRIPTION = '"Norwegian flag morphogenesis (novelty) "\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    NEAT_CONFIG.elitism = 0
    NEAT_CONFIG.do_speciation = False
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=PAIR_SELECTION_F,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )

    NEAT_CONFIG.elitism = 0
    NEAT_CONFIG.do_speciation = True
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=PAIR_SELECTION_F,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )

    NEAT_CONFIG.elitism = 1
    NEAT_CONFIG.do_speciation = False
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=PAIR_SELECTION_F,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )

    NEAT_CONFIG.elitism = 1
    NEAT_CONFIG.do_speciation = True
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=PAIR_SELECTION_F,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )
