from collections import defaultdict
from statistics import median
from typing import List
from uuid import UUID


def get_total_fitnesses_by_species_by_generation(db, scenario_id=1):
    """
    Create a data structure holding information about total fitness for each species for each generation
    """
    scenario = db.get_scenario(scenario_id=scenario_id)
    generations = scenario.generations

    total_fitnesses_by_species_by_generation = []
    for n in range(generations):
        generation = db.get_generation(scenario_id=scenario_id, generation=n)
        fitnesses_by_species = defaultdict(float)
        for individual in generation:
            fitnesses_by_species[individual.genotype.species_id] += individual.fitness

        if fitnesses_by_species:
            total_fitnesses_by_species_by_generation.append(fitnesses_by_species)

    return total_fitnesses_by_species_by_generation


def is_species_stagnant(total_fitnesses_by_species_by_generation: List[dict], species_id: int, stagnation_limit: float):
    """
    Traverse data structure from newest to latest generation to see if the total fitness for the given species
    is stagnant (or going down)
    """
    stagnation_counter = 0

    latest_gen = total_fitnesses_by_species_by_generation[-1]

    median_fitness_for_generation = median(latest_gen.values())

    if latest_gen[species_id] > median_fitness_for_generation:
        return False

    for gen_num in range(len(total_fitnesses_by_species_by_generation) - 1, 0, -1):
        first_gen_index = gen_num
        second_gen_index = gen_num - 1
        fitness_a = total_fitnesses_by_species_by_generation[first_gen_index][species_id]
        fitness_b = total_fitnesses_by_species_by_generation[second_gen_index][species_id]

        if fitness_a > fitness_b:
            break
        else:
            stagnation_counter += 1

    return stagnation_counter >= stagnation_limit


if __name__ == '__main__':
    import os
    from run_experiment import get_db

    STAGNATION_LIMIT = 10

    DB_PATH = os.path.join('sqlite:///', 'db/neat_test/', '2016-10-18 21:07:14.720841.db')
    DB = get_db(DB_PATH)

    total_fitnesses_by_species_by_generation = get_total_fitnesses_by_species_by_generation(DB)

    latest_gen = total_fitnesses_by_species_by_generation[-1]
    current_species = latest_gen.keys()

    for species_id in current_species:
        l = (list(total_fitnesses_by_species_by_generation[x][species_id] for x in
                  range(len(total_fitnesses_by_species_by_generation)))[::-1])[:STAGNATION_LIMIT]
        s = is_species_stagnant(total_fitnesses_by_species_by_generation, species_id, STAGNATION_LIMIT)
        print(UUID(int=species_id).hex, s, '\t', l)
