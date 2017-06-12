import os
from datetime import datetime

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.database import get_db
from ca_neat.ga.selection import sigma_scaled
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.geometry.neighbourhoods import VON_NEUMANN
from ca_neat.patterns.patterns import ALPHABET_4, TRICOLOR
from ca_neat.problems.common import morphogenesis_fitness_f, replication_fitness_f
from ca_neat.run_experiment import initialize_scenario

CA_CONFIG = CAConfig()
CA_CONFIG.alphabet = ALPHABET_4
CA_CONFIG.neighbourhood = VON_NEUMANN
CA_CONFIG.iterations = 30
CA_CONFIG.etc = {
    'pattern': TRICOLOR,
    'wanted_occurrences': 3,
    'penalty_factor': 0.9,
}

NEAT_CONFIG = CPPNNEATConfig()

NEAT_CONFIG.pop_size = 200
NEAT_CONFIG.generations = 1
NEAT_CONFIG.elitism = 1
NEAT_CONFIG.survival_threshold = 0.2
NEAT_CONFIG.stagnation_limit = 15
NEAT_CONFIG.do_speciation = False

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)

PAIR_SELECTION_F = sigma_scaled
FITNESS_F = replication_fitness_f

if __name__ == '__main__':
    NOVELTY_DB_PATH = 'postgresql+psycopg2:///generate_norwegian_flag_find_innovations_2017-05-26T18:38:44.84'

    db = get_db(NOVELTY_DB_PATH)
    session = db.Session()
    scenarios = [1, 2]

    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')

    DB_PATH = 'postgresql+psycopg2:///{}_{}'.format(PROBLEM_NAME, datetime.now().isoformat())

    DESCRIPTION = '"Tricolor replication (novelty) "\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    for scenario in scenarios:
        print(scenario, datetime.now().time().isoformat(), end='\t')
        values = db.get_innovations(scenario_id=scenario, session=session)
        gts = [deserialize_gt(x[0], NEAT_CONFIG) for x in values]
        session.close()

        if not gts:
            continue

        for gt in gts:
            gt.fitness = None

        print('Loaded {} genotypes from innovation archive'.format(len(gts)))

        initialize_scenario(
            db_path=DB_PATH,
            description=DESCRIPTION,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
            initial_genotypes=gts,
        )
