from uuid import UUID

import os
from datetime import datetime

from neat.nn import create_feed_forward_phenotype
from tqdm._tqdm import tqdm

from ca_neat.ca.analysis import calculate_lambda
from ca_neat.config import CAConfig, CPPNNEATConfig
from ca_neat.database import get_db, Scenario, Individual
from ca_neat.ga.population import create_initial_population
from ca_neat.ga.selection import sigma_scaled, random_choice
from ca_neat.ga.serialize import serialize_gt
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

NEAT_CONFIG.pop_size = 1000
NEAT_CONFIG.generations = 100

NEAT_CONFIG.input_nodes = len(CA_CONFIG.neighbourhood)
NEAT_CONFIG.output_nodes = len(CA_CONFIG.alphabet)
NEAT_CONFIG.initial_hidden_nodes = 0

FITNESS_F = morphogenesis_fitness_f

if __name__ == '__main__':
    THIS_FILE = os.path.abspath(__file__)
    PROBLEM_NAME = os.path.split(THIS_FILE)[1].replace('.py', '')

    DB_PATH = 'postgresql+psycopg2:///' + '{}_{}'.format(PROBLEM_NAME, datetime.now().isoformat())

    DESCRIPTION = '"Tricolor flag morphogenesis"\npopulation size: {pop}\ngenerations: {gens}'.format(
        pop=NEAT_CONFIG.pop_size,
        gens=NEAT_CONFIG.generations
    )

    population = list(create_initial_population(NEAT_CONFIG))

    CA_CONFIG.compute_lambda = True
    NEAT_CONFIG.stop_when_optimal_found = False

    NEAT_CONFIG.elitism = 1
    NEAT_CONFIG.do_speciation = True
    NEAT_CONFIG.survival_threshold = 0.2
    NEAT_CONFIG.pair_selection_f = sigma_scaled
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=NEAT_CONFIG.pair_selection_f,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
        initial_genotypes=population,
    )

    NEAT_CONFIG.elitism = 0
    NEAT_CONFIG.do_speciation = True
    NEAT_CONFIG.survival_threshold = 0.2
    NEAT_CONFIG.pair_selection_f = sigma_scaled
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=NEAT_CONFIG.pair_selection_f,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
        initial_genotypes=population,
    )

    NEAT_CONFIG.elitism = 0
    NEAT_CONFIG.do_speciation = False
    NEAT_CONFIG.survival_threshold = 0.2
    NEAT_CONFIG.pair_selection_f = sigma_scaled
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=NEAT_CONFIG.pair_selection_f,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
        initial_genotypes=population,
    )

    NEAT_CONFIG.elitism = 0
    NEAT_CONFIG.do_speciation = False
    NEAT_CONFIG.survival_threshold = 1.0
    NEAT_CONFIG.pair_selection_f = random_choice
    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        fitness_f=FITNESS_F,
        pair_selection_f=NEAT_CONFIG.pair_selection_f,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
        initial_genotypes=population,
    )

    db = get_db(DB_PATH)
    session = db.Session()
    scenario = db.save_scenario(scenario=Scenario(
        description=DESCRIPTION,
        generations=NEAT_CONFIG.generations,
        population_size=NEAT_CONFIG.pop_size
    ), session=session)
    session.commit()

    l = []
    for i, genotype in enumerate(population):
        genotype = genotype.mutate()
        phenotype = create_feed_forward_phenotype(genotype)

        try:
            genotype.fitness = FITNESS_F(phenotype, CA_CONFIG)
            λ = calculate_lambda(cppn=phenotype, ca_config=CA_CONFIG)
        except OverflowError:
            genotype.fitness = 0.0
            λ = 0.0

        individual = Individual(
            scenario_id=scenario.id,
            individual_number=i,
            genotype=serialize_gt(genotype),
            fitness=genotype.fitness,
            generation=0,
            λ=λ,
            species=UUID(int=1),
        )

        l.append(individual)

    session.bulk_save_objects(l)
    session.commit()

    for g in tqdm(range(1, NEAT_CONFIG.generations)):
        l = []
        for i, genotype in enumerate(population):
            genotype = genotype.mutate()
            phenotype = create_feed_forward_phenotype(genotype)

            try:
                genotype.fitness = FITNESS_F(phenotype, CA_CONFIG)
                λ = calculate_lambda(cppn=phenotype, ca_config=CA_CONFIG)
            except OverflowError:
                genotype.fitness = 0.0
                λ = 0.0

            individual = Individual(
                scenario_id=scenario.id,
                individual_number=i,
                genotype=serialize_gt(genotype),
                fitness=genotype.fitness,
                generation=g,
                λ=λ,
                species=UUID(int=1),
            )

            l.append(individual)

        session.bulk_save_objects(l)
        session.commit()
