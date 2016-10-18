import os
from datetime import datetime

from neat.config import Config
from neat.genes import ConnectionGene, NodeGene
from neat.genome import Genome

from geometry.neighbourhoods import VON_NEUMANN
from run_experiment import initialize_scenario
from run_neat import create_initial_population
from selection import sigma_scaled
from visualization.plot_fitness import plot_fitnesses_over_generations


def fitness(phenotype, **kwargs):
    from geometry.cell_grid import FiniteCellGrid2D
    from ca.iterate import n_iterations

    neighbourhood = kwargs.get('neighbourhood')
    alphabet = kwargs.get('alphabet')

    r = 5

    initial = FiniteCellGrid2D(
        cell_states=alphabet,
        neighbourhood=neighbourhood,
        x_range=(-r, r),
        y_range=(-r, r),
        values={(0, 0): 1}
    )

    from utils import make_step_f
    step = make_step_f(0.5)

    def transition_f(args):
        t = tuple(int(x) for x in args)
        return step(phenotype.serial_activate(t)[0])

    grid = n_iterations(initial_grid=initial, transition_f=transition_f, n=10)

    return sum(sum(int(x) for x in row) for row in grid.get_whole()) / grid.area


def geno_to_pheno_f(genotype, **kwargs):
    from neat.nn import create_feed_forward_phenotype
    return create_feed_forward_phenotype(genotype)


def mutation_f(genotype, **kwargs):
    genotype.mutate()
    return genotype


def crossover(a, b, **kwargs):
    from uuid import uuid4
    return a.crossover(other=b, child_id=uuid4())


if __name__ == '__main__':
    ALPHABET = (0, 1)
    NEIGHBOURHOOD = VON_NEUMANN
    POPULATION_SIZE = 100
    N_GENERATIONS = 100

    NEAT_CONFIG = Config()
    NEAT_CONFIG.pop_size = POPULATION_SIZE
    NEAT_CONFIG.input_nodes = len(NEIGHBOURHOOD)
    NEAT_CONFIG.output_nodes = 1
    NEAT_CONFIG.node_gene_type = NodeGene
    NEAT_CONFIG.conn_gene_type = ConnectionGene
    NEAT_CONFIG.activation_functions = ('sigmoid',)
    NEAT_CONFIG.weight_stdev = 1.0
    NEAT_CONFIG.genotype = Genome
    NEAT_CONFIG.compatibility_threshold = 3.0
    NEAT_CONFIG.prob_add_conn = 0.988
    NEAT_CONFIG.prob_add_node = 0.085
    NEAT_CONFIG.prob_delete_conn = 0.146
    NEAT_CONFIG.prob_delete_node = 0.0352
    NEAT_CONFIG.prob_mutate_bias = 0.0509
    NEAT_CONFIG.bias_mutation_power = 2.093
    NEAT_CONFIG.prob_mutate_response = 0.1
    NEAT_CONFIG.response_mutation_power = 0.1
    NEAT_CONFIG.prob_mutate_weight = 0.460
    NEAT_CONFIG.prob_replace_weight = 0.0245
    NEAT_CONFIG.weight_mutation_power = 0.825
    NEAT_CONFIG.prob_mutate_activation = 0.0
    NEAT_CONFIG.prob_toggle_link = 0.0138
    NEAT_CONFIG.max_weight = 30
    NEAT_CONFIG.min_weight = -30
    NEAT_CONFIG.excess_coefficient = 1.0
    NEAT_CONFIG.disjoint_coefficient = 1.0
    NEAT_CONFIG.weight_coefficient = 0.4

    INITIAL_GENOTYPES = list(create_initial_population(NEAT_CONFIG))

    MUTATION_CHANCE = 0.25
    GENO_TO_PHENO_F = geno_to_pheno_f
    FITNESS_F = fitness
    SELECTION_F = sigma_scaled
    CROSSOVER_F = crossover
    MUTATION_F = mutation_f

    DB_DIR = 'db/neat_test/'
    DB_PATH = os.path.join('sqlite:///', DB_DIR, '{}.db'.format(datetime.now()))

    DESCRIPTION = '"Max one 2D CA"\npopulation size: {}\ngenerations: {}'.format(POPULATION_SIZE, N_GENERATIONS)
    gt = INITIAL_GENOTYPES[0]

    initialize_scenario(
        db_path=DB_PATH,
        description=DESCRIPTION,
        initial_genotypes=INITIAL_GENOTYPES,
        population_size=POPULATION_SIZE,
        generations=N_GENERATIONS,
        geno_to_pheno_f=GENO_TO_PHENO_F,
        fitness_f=FITNESS_F,
        selection_f=SELECTION_F,
        crossover_f=CROSSOVER_F,
        mutation_f=MUTATION_F,
        mutation_chance=MUTATION_CHANCE,
        alphabet=ALPHABET,
        neighbourhood=NEIGHBOURHOOD,
        elitism=2,
        stagnation_limit=None,
        survival_threshold=0.5,
        compatibility_threshold=3.0,
    )
    plot_fitnesses_over_generations(DB_PATH, title=DESCRIPTION, interval=1)
