import os

from database import Individual, get_db
from run_experiment import initialize_generation
from run_neat import sort_into_species
from utils import PROJECT_ROOT, pluck

if __name__ == '__main__':
    problem_name = 'generate_border'
    db_file = '2016-11-23 13:51:24.771958.db'
    from problems.generate_border import CA_CONFIG, NEAT_CONFIG, FITNESS_F, PAIR_SELECTION_F

    db_path = 'sqlite:///{}'.format(os.path.join(PROJECT_ROOT, 'problems', 'results', problem_name, db_file))
    db = get_db(db_path)

    session = db.Session()

    for l in """""".strip().split('\n'):
        l = l.strip()

        if 'True' in l:
            continue

        scenario_id, generation_n, _ = l.split(' ')
        scenario_id = int(scenario_id)
        generation_n = int(generation_n)

        if generation_n >= 500:
            continue

        print(scenario_id, generation_n)

        generation_n -= 1
        genotypes = list(pluck(db.get_generation(scenario_id, generation=generation_n), 'genotype'))

        assert genotypes

        to_purge = db.get_individuals(scenario_id).filter(Individual.generation >= generation_n)
        print(scenario_id, 'deleting', to_purge.count())
        to_purge.delete()

        session.commit()

        species = sort_into_species(genotypes)

        initialize_generation(
            db_path=db_path,
            scenario_id=scenario_id,
            generation=generation_n,
            genotypes=genotypes,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
