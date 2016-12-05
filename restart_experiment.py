import os

from database import Individual, get_db
from run_experiment import initialize_generation
from run_neat import sort_into_species
from utils import PROJECT_ROOT, pluck

if __name__ == '__main__':
    problem_name = 'generate_tricolor'
    db_file = '2016-12-04 18:09:12.299816.db'
    from problems.generate_tricolor import CA_CONFIG, NEAT_CONFIG, FITNESS_F, PAIR_SELECTION_F

    db_path = 'sqlite:///{}'.format(os.path.join(PROJECT_ROOT, 'problems', 'results', problem_name, db_file))
    db = get_db(db_path)

    session = db.Session()

    for scenario_id, generation_n in [
        (5, 5),
        (49, 470),
    ]:
        genotypes = list(pluck(db.get_generation(scenario_id, generation=generation_n), 'genotype'))

        assert genotypes

        to_purge = db.get_individuals(scenario_id).filter(Individual.generation >= generation_n)
        print('deleting', to_purge.count())
        to_purge.delete()

        session.commit()

        species = sort_into_species(genotypes)

        assert len(species) > 1

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
