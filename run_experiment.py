import logging
from typing import List

from celery.app import shared_task
from neat.genome import Genome
from neat.nn import create_feed_forward_phenotype
from neat.population import CompleteExtinctionException
from sqlalchemy.sql.functions import now

from add_dill import add_dill
from config import CAConfig, CPPNNEATConfig
from database import Individual, Scenario, get_db
from run_neat import create_initial_population, neat_reproduction, sort_into_species, speciate
from stagnation import get_total_fitnesses_by_species_by_generation, is_species_stagnant

add_dill()


def initialize_scenario(db_path: str, description: str, fitness_f, pair_selection_f,
                        neat_config: CPPNNEATConfig, ca_config: CAConfig):
    db = get_db(db_path)
    session = db.Session()
    scenario = db.save_scenario(scenario=Scenario(
        description=description,
        generations=neat_config.generations,
        population_size=neat_config.pop_size
    ), session=session)

    assert scenario

    initial_genotypes = list(create_initial_population(neat_config))

    species = speciate(initial_genotypes, compatibility_threshold=neat_config.compatibility_threshold)

    assert len(species) > 1

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario.id,
        generation=0,
        genotypes=initial_genotypes,
        fitness_f=fitness_f,
        pair_selection_f=pair_selection_f,
        neat_config=neat_config,
        ca_config=ca_config,
    )

    session.flush()


def initialize_generation(db_path: str, scenario_id: int, generation: int, genotypes: List[Genome],
                          pair_selection_f, fitness_f, neat_config: CPPNNEATConfig, ca_config: CAConfig):
    from celery import group, chord

    grouped_tasks = group(handle_individual.s(
        db_path=db_path,
        scenario_id=scenario_id,
        generation=generation,
        individual_number=i,
        genotype=genotype,
        fitness_f=fitness_f,
        neat_config=neat_config,
        ca_config=ca_config,
    ) for i, genotype in enumerate(genotypes))

    final_task = finalize_generation.subtask(
        args=(db_path, scenario_id, generation, fitness_f, pair_selection_f, neat_config, ca_config)
    )

    chord(grouped_tasks, final_task)()


@shared_task(name='finalize_generation', bind=True)
def finalize_generation(task, results, db_path: str, scenario_id: int, generation_n: int, fitness_f, pair_selection_f,
                        neat_config: CPPNNEATConfig, ca_config: CAConfig):
    """
    'results' is return values of preceding group of tasks, can safely be ignored
    """
    db = get_db(db_path)
    session = db.Session()
    scenario = db.get_scenario(scenario_id, session=session)
    population = db.get_generation(scenario_id, generation_n, session=session)

    next_gen = generation_n + 1

    optimal_solution = population.filter(Individual.fitness >= 1.0)
    optimal_found = (neat_config.stop_when_optimal_found and session.query(optimal_solution.exists()).scalar())

    if (next_gen == scenario.generations) or optimal_found:
        logging.info('Scenario {} finished after {} generations'.format(scenario_id, next_gen))
        return

    species = sort_into_species([individual.genotype for individual in population])

    stagnation_limit = neat_config.stagnation_limit
    if (not stagnation_limit) or (generation_n < stagnation_limit):
        alive_species = species
    else:
        total_fitnesses_by_species_by_generation = get_total_fitnesses_by_species_by_generation(
            db=db,
            scenario_id=scenario_id,
            generation_range=(generation_n - stagnation_limit, next_gen),
        )
        alive_species = [s for s in species if not is_species_stagnant(
            total_fitnesses_by_species_by_generation=total_fitnesses_by_species_by_generation,
            species_id=s.ID,
            stagnation_limit=stagnation_limit,
            median_threshold=neat_config.stagnation_median_threshold,
        )]

    if not alive_species:
        raise CompleteExtinctionException

    new_species, new_genotypes = neat_reproduction(
        species=alive_species,
        pop_size=scenario.population_size,
        survival_threshold=neat_config.survival_threshold,
        elitism=neat_config.elitism,
        pair_selection_f=pair_selection_f,
    )

    species = speciate(
        new_genotypes,
        compatibility_threshold=neat_config.compatibility_threshold,
        existing_species=new_species
    )

    initialize_generation(
        db_path=db_path,
        scenario_id=scenario_id,
        generation=next_gen,
        genotypes=new_genotypes,
        fitness_f=fitness_f,
        pair_selection_f=pair_selection_f,
        neat_config=neat_config,
        ca_config=ca_config,
    )

    logging.info('Finished generation {gen} of scenario {scen}: {pop} individuals / {species} species'.format(
        gen=generation_n,
        scen=scenario_id,
        pop=len(new_genotypes),
        species=len(species),
    ))
    return generation_n


@shared_task(name='handle_individual')
def handle_individual(db_path: str, scenario_id: int, generation: int, individual_number: int, genotype: Genome,
                      fitness_f, neat_config: CPPNNEATConfig, ca_config: CAConfig):
    phenotype = create_feed_forward_phenotype(genotype)
    fitness = fitness_f(phenotype=phenotype, ca_config=ca_config)

    assert 0.0 <= fitness <= 1.0

    if hasattr(genotype, 'fitness'):
        genotype.fitness = fitness

    individual = Individual(
        scenario_id=scenario_id,
        individual_number=individual_number,
        genotype=genotype,
        fitness=fitness,
        generation=generation,
        timestamp=now()
    )

    get_db(db_path).save_individual(individual)

    return 'Scenario {}, generation {}, individual {}, fitness {}'.format(
        scenario_id,
        generation,
        individual_number,
        fitness
    )
