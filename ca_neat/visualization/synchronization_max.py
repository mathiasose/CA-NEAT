import os
from collections import defaultdict
from operator import itemgetter
from random import uniform
import os
from datetime import datetime
from random import uniform

from neat.nn import FeedForwardNetwork
from tqdm._tqdm import tqdm

from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.ga.selection import sigma_scaled
from ca_neat.geometry.neighbourhoods import LLLCRRR
from ca_neat.patterns.patterns import ALPHABET_2
from ca_neat.problems.majority import create_binary_pattern
from ca_neat.run_experiment import initialize_scenario
from pandas import json
from statistics import mode, mean
from typing import Iterator
from typing import Sequence

from math import exp
from neat.nn import FeedForwardNetwork, create_feed_forward_phenotype

from ca_neat.ca.analysis import serialize_cppn_rule
from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.geometry.cell_grid import CELL_STATE_T
from ca_neat.geometry.cell_grid import FiniteCellGrid1D, ToroidalCellGrid1D
from ca_neat.problems.majority import create_binary_pattern
from ca_neat.problems.synchronization import CA_CONFIG, NEAT_CONFIG, fitness_f
from ca_neat.utils import create_state_normalization_rules, invert_pattern, random_string
from operator import itemgetter
from typing import Iterator, Sequence
from neat.nn import FeedForwardNetwork
from ca_neat.ca.iterate import iterate_ca_n_times_or_until_cycle_found
from ca_neat.utils import create_state_normalization_rules
from ca_neat.geometry.cell_grid import ToroidalCellGrid1D, CELL_STATE_T
from random import shuffle

if __name__ == '__main__':
    DB_PATH = 'postgresql+psycopg2:///synchronization_2017-05-06T17:00:23.012278'
    db = get_db(DB_PATH)

    session = db.Session()

    assert session.query(Individual).count()

    scenarios = list(db.get_scenarios(session))

    maxs = defaultdict(list)
    for scenario in tqdm(scenarios):
        xs = db.get_individuals(scenario_id=scenario.id, session=session) \
            .order_by(-Individual.fitness)

        if not xs.count():
            continue

        for g in range(100):
            maxs[scenario.id].append(xs.filter(Individual.generation == g)[0].fitness)

    import os
    from statistics import mean, median

    import matplotlib.pyplot as plt
    import seaborn
    from sqlalchemy.sql.functions import func
    from tqdm._tqdm import tqdm

    from ca_neat.database import Db, Individual, Scenario, get_db
    from ca_neat.utils import PROJECT_ROOT

    seaborn.set_style('whitegrid')

    fig = plt.figure()

    for sid in sorted(maxs.keys()):
        plt.plot(maxs[sid])

    plt.show(block=True)
