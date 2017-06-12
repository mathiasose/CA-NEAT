from statistics import mode

from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import exists
from sqlalchemy.sql.functions import func

from ca_neat.database import Individual, get_db
from ca_neat.ga.population import sort_into_species
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.run_experiment import initialize_generation
from ca_neat.utils import pluck

S = -1
FROM_GENERATION = None  # 99

if __name__ == '__main__':
    from ca_neat.problems.morphogenesis.generate_border_with_coord_input import CA_CONFIG, NEAT_CONFIG, \
        FITNESS_F, PAIR_SELECTION_F

    DB_PATH = 'postgresql+psycopg2:///generate_border_with_coord_input_2017-05-25T14:11:14.401192'

    # blob = '/home/ose/CA-NEAT/ca_neat/problems/synchronization_2017-05-06T17:00:23.012278.json'
    # with open(blob, 'r') as f:
    #     import json
    #     CA_CONFIG.etc = json.load(f)

    DB = get_db(DB_PATH)

    scenario_ids = sorted([s.id for s in DB.get_scenarios()])

    for scenario_id in scenario_ids:
        try:
            session = DB.Session()
            scenario = DB.get_scenario(scenario_id=scenario_id, session=session)

            done = session.query(
                exists()
                    .where(Individual.scenario_id == scenario.id)
                    .where(Individual.fitness >= 1.0)
            ).scalar()

            if done:
                print(scenario.id, 'done')
                continue

            d = session.query(func.max(Individual.generation)) \
                .filter(Individual.scenario_id == scenario.id) \
                .scalar()

            if d is None:
                print(scenario.id, 'no data?')
                continue

            if FROM_GENERATION:
                generation_n = FROM_GENERATION
            else:
                generation_n = d

            if generation_n >= (scenario.generations - 1):
                print(scenario.id, 'no more generations')
                continue

            while True:
                assert generation_n >= 0

                generation = DB.get_generation(scenario_id=scenario.id, generation=generation_n, session=session)
                print(scenario.id, generation_n, generation.count())

                if generation.count() < (0.95 * NEAT_CONFIG.pop_size):
                    generation_n -= 1
                    continue

                genotypes = list(pluck(generation, 'genotype'))

                if any((g is None) for g in genotypes):
                    generation_n -= 1
                    continue

                break

            to_purge = DB.get_individuals(scenario.id, session=session) \
                .filter(Individual.generation >= generation_n)

            print(scenario_id, 'deleting', to_purge.count())

            to_purge.delete()

            session.commit()
        except OperationalError as e:
            print(scenario_id, 'DB locked?')
            print(e)
            continue
        finally:
            session.close()

        genotypes = list(
            deserialize_gt(
                gt_json_bytes=gt,
                neat_config=NEAT_CONFIG
            ) for gt in genotypes
        )

        species = sort_into_species(genotypes)

        initialize_generation(
            db_path=DB_PATH,
            scenario_id=scenario_id,
            generation=generation_n,
            genotypes=genotypes,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
