import copy

import graphviz

OUTPUT_NODE_STYLE = {'style': 'filled'}
INPUT_NODE_STYLE = {'style': 'filled', 'shape': 'box'}
INVISIBLE_NODE = {'style': 'invisible', }

SYMBOLS = {
    'sigmoid': 'σ'
}


def symbol(name):
    return SYMBOLS.get(name, name)


def draw_net(genome, view=False, filename=None, show_disabled=True, prune_unused=False, fmt='svg'):
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
            label = symbol(ng.activation_type)
            input_attrs = INPUT_NODE_STYLE
            input_attrs['fillcolor'] = 'lightgray'
            in_hidden_node = str(ng_id) + '_in'
            node_id = str(ng_id)
            dot.node(in_hidden_node, label=str(ng_id), _attributes=INVISIBLE_NODE)
            dot.node(node_id, label=label, _attributes=input_attrs)
            dot.edge(in_hidden_node, node_id)

    outputs = set()
    for ng_id, ng in genome.node_genes.items():
        if ng.type == 'OUTPUT':
            outputs.add(ng_id)
            label = symbol(ng.activation_type)
            node_attrs = OUTPUT_NODE_STYLE
            node_attrs['fillcolor'] = 'lightblue'

            out_hidden_node = str(ng_id) + '_out'
            node_id = str(ng_id)
            sg = graphviz.Digraph()
            sg.node(node_id, label=label, _attributes=node_attrs)
            sg.node(out_hidden_node, label=str(ng_id), _attributes=INVISIBLE_NODE)
            sg.edge(node_id, out_hidden_node)
            dot.subgraph(sg)

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

        label = symbol(genome.node_genes[n].activation_type)
        attrs = {'style': 'filled'}
        attrs['fillcolor'] = 'white'
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
            edge_attrs = {'style': style, 'color': color, 'penwidth': width}
            dot.edge(a, b, _attributes=edge_attrs, label=str(round(cg.weight, 1)))

    dot.render(filename, view=view)

    return dot
