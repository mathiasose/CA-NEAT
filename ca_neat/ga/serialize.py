import json
from uuid import UUID

from neat.genes import ConnectionGene, NodeGene
from neat.genome import Genome

from ca_neat.config import CPPNNEATConfig
from ca_neat.ga.population import create_initial_population


def serialize_gt(gt: Genome) -> bytes:
    d = {
        'ID': gt.ID,
        'parent1_id': gt.parent1_id,
        'parent2_id': gt.parent2_id,
        'species_id': gt.species_id,
        'fitness': gt.fitness,
        'node_genes': [
            {
                'ID': ng.ID,
                'type': ng.type,
                'bias': ng.bias,
                'response': ng.response,
                'activation_type': ng.activation_type,
            } for ng in gt.node_genes.values()
            ],
        'conn_genes': [
            {
                'in_node_id': cg.in_node_id,
                'out_node_id': cg.out_node_id,
                'weight': cg.weight,
                'enabled': cg.enabled,
            } for cg in gt.conn_genes.values()
            ],
    }

    return json.dumps(d).encode()


def deserialize_gt(gt_json_bytes: bytes, neat_config: CPPNNEATConfig) -> Genome:
    gt_dict = json.loads(gt_json_bytes, encoding='ascii')

    gt = Genome(
        ID=gt_dict['ID'],
        config=neat_config,
        parent1_id=gt_dict['parent1_id'],
        parent2_id=gt_dict['parent2_id'],
    )
    gt.species_id = gt_dict['species_id']
    gt.fitness = gt_dict['fitness']

    gt.node_genes = {
        ng_dict['ID']: NodeGene(
            ID=ng_dict['ID'],
            node_type=ng_dict['type'],
            bias=float(ng_dict['bias']),
            response=float(ng_dict['response']),
            activation_type=ng_dict['activation_type'],
        ) for ng_dict in gt_dict['node_genes']
        }

    gt.conn_genes = {
        (cg_dict['in_node_id'], cg_dict['out_node_id']): ConnectionGene(
            in_node_id=cg_dict['in_node_id'],
            out_node_id=cg_dict['out_node_id'],
            weight=float(cg_dict['weight']),
            enabled=cg_dict['enabled'],
        ) for cg_dict in gt_dict['conn_genes']
        }

    return gt


if __name__ == '__main__':
    from ca_neat.problems.morphogenesis.generate_border import NEAT_CONFIG

    neat_config = NEAT_CONFIG
    gt = next(create_initial_population(neat_config=neat_config))

    serialized = serialize_gt(gt)
    print(serialized)
    print(type(serialized))

    ds = deserialize_gt(serialized, neat_config)
    print(ds)
