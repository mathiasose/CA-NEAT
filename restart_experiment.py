import os

from database import get_db, Individual
from run_experiment import initialize_generation
from run_neat import sort_into_species
from utils import PROJECT_ROOT, pluck

if __name__ == '__main__':
    problem_name = 'replicate_norwegian_flag'
    db_file = '2016-11-11 20:53:43.950038.db'
    scenario_id = 3
    generation_n = 10

    from problems.replicate_norwegian_flag import CA_CONFIG, NEAT_CONFIG, fitness_f, PAIR_SELECTION_F

    db_path = 'sqlite:///{}'.format(os.path.join(PROJECT_ROOT, 'problems', 'results', problem_name, db_file))
    db = get_db(db_path)

    session = db.Session()
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
        fitness_f=fitness_f,
        pair_selection_f=PAIR_SELECTION_F,
        neat_config=NEAT_CONFIG,
        ca_config=CA_CONFIG,
    )
