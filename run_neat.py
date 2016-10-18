import math
from typing import List
from uuid import uuid4

import random
from matplotlib.colors import ListedColormap
from neat.config import Config
from neat.genes import ConnectionGene, NodeGene
from neat.genome import Genome
from neat.nn import create_feed_forward_phenotype
from neat.species import Species


def create_initial_population(neat_config):
    for _ in range(neat_config.pop_size):
        g_id = uuid4()
        g = neat_config.genotype.create_unconnected(g_id, neat_config)
        g.connect_full()
        yield g


def speciate(genotypes: List[Genome], compatibility_threshold, existing_species=None):
    species = []
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


def neat_development(genotype, **kwargs):
    return create_feed_forward_phenotype(genotype)


def neat_mutation(genotype, **kwargs):
    genotype.mutate()
    return genotype


def neat_crossover(a, b, **kwargs):
    return a.crossover(other=b, child_id=uuid4())


def sort_into_species(genotypes: List[Genome]):
    species = {}
    for gt in genotypes:
        species_id = gt.species_id

        assert species_id is not None

        try:
            species[species_id].add(gt)
        except KeyError:
            species[species_id] = Species(gt, species_id)

    return species.values()


def neat_reproduction(species: List[Species], pop_size, survival_threshold, elitism=0, **kwargs):
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
        spawn = len(s.members)
        if sfitness > avg_adjusted_fitness:
            spawn *= 1.1
        else:
            spawn *= 0.9
        spawn_amounts.append(spawn)

    # Normalize the spawn amounts so that the next generation is roughly
    # the population size requested by the user.
    total_spawn = sum(spawn_amounts)
    norm = pop_size / total_spawn
    spawn_amounts = [int(round(n * norm)) for n in spawn_amounts]

    new_population = []
    new_species = []
    for spawn, (s, sfitness) in zip(spawn_amounts, species_fitness):
        # If elitism is enabled, each species always at least gets to retain its elites.
        spawn = max(spawn, elitism)

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
            new_population.extend(old_members[:elitism])
            spawn -= elitism

        if spawn <= 0:
            continue

        # Only use the survival threshold fraction to use as parents for the next generation.
        repro_cutoff = int(math.ceil(survival_threshold * len(old_members)))
        # Use at least two parents no matter what the threshold fraction result is.
        repro_cutoff = max(repro_cutoff, 2)
        old_members = old_members[:repro_cutoff]

        # Randomly choose parents and produce the number of offspring allotted to the species.
        while spawn > 0:
            spawn -= 1

            parent1 = random.choice(old_members)
            parent2 = random.choice(old_members)

            # Note that if the parents are not distinct, crossover will produce a
            # genetically identical clone of the parent (but with a different ID).
            child = parent1.crossover(parent2, uuid4().int)
            new_population.append(child.mutate())

    return new_species, new_population


if __name__ == '__main__':
    test_config = Config()
    test_config.input_nodes = 2
    test_config.output_nodes = 1
    test_config.node_gene_type = NodeGene
    test_config.conn_gene_type = ConnectionGene
    test_config.activation_functions = ('sigmoid',)
    test_config.weight_stdev = 1.0
    test_config.pop_size = 10
    test_config.genotype = Genome
    test_config.compatibility_threshold = 3.0
    test_config.prob_add_conn = 0.988
    test_config.prob_add_node = 0.085
    test_config.prob_delete_conn = 0.146
    test_config.prob_delete_node = 0.0352
    test_config.prob_mutate_bias = 0.0509
    test_config.bias_mutation_power = 2.093
    test_config.prob_mutate_response = 0.1
    test_config.response_mutation_power = 0.1
    test_config.prob_mutate_weight = 0.460
    test_config.prob_replace_weight = 0.0245
    test_config.weight_mutation_power = 0.825
    test_config.prob_mutate_activation = 0.0
    test_config.prob_toggle_link = 0.0138

    test_config.max_weight = 30
    test_config.min_weight = -30

    if __name__ == '__main__':

        # config.reproduction_type = DefaultReproduction
        # config.stagnation_type = DefaultStagnation


        population = list(create_initial_population(test_config))
        # print(*population, sep='\n\n')
        species = list(speciate(population))
        # print(*species, sep='\n')
        # print(
        #    *(create_feed_forward_phenotype(genotype).serial_activate(inputs=[0, 1]) for genotype in population),
        #    sep='\n'
        # )

        import matplotlib.pyplot as plt
        import seaborn

        seaborn.set(style='white')

        gt = population[0]
        for _ in range(10):
            gt.mutate()

            pt = create_feed_forward_phenotype(gt)
            f = lambda *args: pt.serial_activate(args)[0]
            R = 50
            mat = [[f(x, y) for x in range(-R, R)] for y in range(-R, R)]
            # print(*mat, sep='\n')


            plt.imshow(mat, interpolation='nearest', extent=(-R, R, -R, R),
                       cmap=ListedColormap(seaborn.color_palette("hls")))
            plt.show()
