import copy

import graphviz

OUTPUT_NODE_STYLE = {'style': 'filled'}

INPUT_NODE_STYLE = {'style': 'filled', 'shape': 'box'}


def draw_net(genome, view=False, filename=None, node_labels=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    if node_labels is None:
        node_labels = {}

    assert type(node_labels) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for ng_id, ng in genome.node_genes.items():
        if ng.type == 'INPUT':
            inputs.add(ng_id)
            label = node_labels.get(ng_id, str(ng_id))
            input_attrs = INPUT_NODE_STYLE
            input_attrs['fillcolor'] = node_colors.get(ng_id, 'lightgray')
            dot.node(str(ng_id), label=label, _attributes=input_attrs)

    outputs = set()
    for ng_id, ng in genome.node_genes.items():
        if ng.type == 'OUTPUT':
            outputs.add(ng_id)
            label = node_labels.get(ng_id, str(ng_id))
            node_attrs = OUTPUT_NODE_STYLE
            node_attrs['fillcolor'] = node_colors.get(ng_id, 'lightblue')

            dot.node(str(ng_id), label=label, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.conn_genes.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.node_genes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        label = node_labels.get(n, str(n))
        attrs = {'style': 'filled'}
        attrs['fillcolor'] = node_colors.get(str(n), 'white')
        dot.node(str(n), label=label, _attributes=attrs)

    for cg in genome.conn_genes.values():
        if cg.enabled or show_disabled:
            if cg.in_node_id not in used_nodes or cg.out_node_id not in used_nodes:
                continue

            a = str(cg.in_node_id)
            b = str(cg.out_node_id)
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
