import os
from datetime import datetime

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.database import get_db
from ca_neat.ga.selection import sigma_scaled
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.geometry.neighbourhoods import VON_NEUMANN
from ca_neat.patterns.patterns import ALPHABET_2, BORDER, SEED_6X6
from ca_neat.problems.common import morphogenesis_fitness_f
from ca_neat.run_experiment import initialize_scenario

CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_2
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 30
CA_CONFIG.etc = {
    'target_pattern': BORDER,
    'seed': SEED_6X6,
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 100
NEAT_CONFIG.elitism = 1
NEAT_CONFIG.survival_threshold = 0.2
NEAT_CONFIG.stagnation_limit = 15

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)

PAIR_SELECTION_F = sigma_scaled
FITNESS_F = morphogenesis_fitness_f

if __name__ == '__main__':
    NOVELTY_DB_PATH = 'postgresql+psycopg2:///generate_border_find_innovations_novelty_2017-04-21T19:15:50.286187'
    db = get_db(NOVELTY_DB_PATH)
    session = db.Session()
    values = db.get_innovations(scenario_id=1, session=session)
    gts = [deserialize_gt(x[0], NEAT_CONFIG) for x in values]
    session.close()

    for gt in gts:
        gt.fitness = None

    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')

    DB_PATH = 'postgresql+psycopg2:///{}_{}'.format(PROBLEM_NAME, datetime.now().isoformat())

    DESCRIPTION = '"Border morphogenesis"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for _ in range(1):
        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
            initial_genotypes=gts,
        )
