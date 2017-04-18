import math
import random
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from uuid import uuid4

from neat.genome import Genome
from neat.species import Species

from ca_neat.config import CPPNNEATConfig
from ca_neat.ga.selection import PAIR_SELECTION_F_T, TooFewIndividuals, random_choice


def create_initial_population(neat_config: CPPNNEATConfig) -> Iterator[Genome]:
    for _ in range(neat_config.pop_size):
        g_id = uuid4().int
        g = neat_config.genotype.create_unconnected(g_id, neat_config)

        hidden_nodes = neat_config.initial_hidden_nodes
        if hidden_nodes:
            g.add_hidden_nodes(hidden_nodes)

        if neat_config.initial_connection == 'fs_neat':
            g.connect_fs_neat()
        elif neat_config.initial_connection == 'fully_connected':
            g.connect_full()
        elif neat_config.initial_connection == 'partial':
            if callable(neat_config.connection_fraction):
                fraction = neat_config.connection_fraction()
            else:
                fraction = neat_config.connection_fraction

            g.connect_partial(fraction)

        yield g


def speciate(genotypes: Sequence[Genome], compatibility_threshold: float,
             existing_species: Optional[List[Species]] = None) -> List[Species]:
    species = []  # type: List[Species]
    if isinstance(existing_species, list):
        species += existing_species

    for individual in genotypes:
        # Find the species with the most similar representative.
        min_distance = None
        closest_species = None
        for s in species:
            distance = individual.distance(s.representative)
            if distance < compatibility_threshold \
                    and (min_distance is None or distance < min_distance):
                closest_species = s
                min_distance = distance

        if closest_species:
            closest_species.add(individual)
        else:
            # No species is similar enough, create a new species for this individual.
            species.append(Species(individual, uuid4().int))

    # Only keep non-empty species.
    species = [s for s in species if s.members]

    # Select a random current member as the new representative.
    for s in species:
        s.representative = random.choice(s.members)

    return species


def sort_into_species(genotypes: Iterable[Genome]) -> Iterable[Species]:
    species: Dict[int, Species] = {}
    for gt in genotypes:
        species_id = gt.species_id

        assert species_id is not None

        try:
            species[species_id].add(gt)
        except KeyError:
            species[species_id] = Species(gt, species_id)

    return set(species.values())


def neat_reproduction(species: Iterable[Species], pop_size: int, survival_threshold: float,
                      pair_selection_f: PAIR_SELECTION_F_T, elitism: int = 0, **kwargs) \
        -> Tuple[List[Species], List[Genome]]:
    species_fitness = []
    avg_adjusted_fitness = 0.0
    for s in species:
        species_sum = 0.0
        for m in s.members:
            af = m.fitness / len(s.members)
            species_sum += af

        sfitness = species_sum / len(s.members)
        species_fitness.append((s, sfitness))
        avg_adjusted_fitness += sfitness

    avg_adjusted_fitness /= len(species_fitness)

    # Compute the number of new individuals to create for the new generation.
    spawn_amounts = []
    for s, sfitness in species_fitness:
        spawn = float(len(s.members))
        if sfitness > avg_adjusted_fitness:
            spawn *= 1.1
        else:
            spawn *= 0.9
        spawn_amounts.append(spawn)

    total_spawn = sum(spawn_amounts)
    norm = pop_size / total_spawn
    spawn_amounts = [int(round(n * norm)) for n in spawn_amounts]

    new_population: List[Genome] = []
    new_species = []
    for spawn, (s, sfitness) in zip(spawn_amounts, species_fitness):
        if spawn <= 0:
            continue

        # The species has at least one member for the next generation, so retain it.
        old_members = s.members
        s.members = []
        new_species.append(s)

        # Sort members in order of descending fitness.
        old_members.sort(reverse=True)

        # Transfer elites to new generation.
        if elitism > 0:
            elites = old_members[:min(elitism, spawn)]
            new_population.extend(elites)
            spawn -= len(elites)

        if spawn <= 0:
            continue

        # Only use the survival threshold fraction to use as parents for the next generation.
        repro_cutoff = int(math.ceil(survival_threshold * len(old_members)))
        # Use at least two parents no matter what the threshold fraction result is.
        repro_cutoff = max(repro_cutoff, 2)
        old_members = old_members[:repro_cutoff]

        # choose parents and produce the number of offspring allotted to the species.
        try:
            pair_generator = pair_selection_f(old_members)
        except TooFewIndividuals:
            pair_generator = random_choice(old_members)

        while spawn > 0:
            spawn -= 1

            parent1, parent2 = next(pair_generator)

            # Note that if the parents are not distinct, crossover will produce a
            # genetically identical clone of the parent (but with a different ID).
            child = parent1.crossover(parent2, uuid4().int)
            new_population.append(child.mutate())

    return new_species, new_population
