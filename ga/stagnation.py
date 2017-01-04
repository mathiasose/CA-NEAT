from statistics import mean
from typing import List


def is_species_stagnant(fitnesses_by_species_by_generation: List[dict], species_id: int, stagnation_limit: float,
                        f=mean):
    """
    Traverse data structure from newest to latest generation to see if the total fitness for the given species
    is stagnant (or going down)
    """
    stagnation_counter = 0

    for gen_num in range(len(fitnesses_by_species_by_generation) - 1, 0, -1):
        fitness_last_gen = f(fitnesses_by_species_by_generation[gen_num].get(species_id, [0]))
        fitness_prev_gen = f(fitnesses_by_species_by_generation[gen_num - 1].get(species_id, [0]))

        if fitness_last_gen > fitness_prev_gen:
            break
        else:
            stagnation_counter += 1

    return stagnation_counter >= stagnation_limit
