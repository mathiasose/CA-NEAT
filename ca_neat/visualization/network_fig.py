import graphviz
from neat.genome import Genome

REGULAR_NODE_STYLE = {'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2', 'style': 'filled',
                      'fillcolor': 'white', }
OUTPUT_NODE_STYLE = {'style': 'filled', 'fillcolor': 'lightblue', }
INPUT_NODE_STYLE = {'style': 'filled', 'shape': 'box', 'fillcolor': 'lightgray', }
INVISIBLE_NODE = {'style': 'dotted', }
IO_EDGE_COLOR = 'blue'

SYMBOLS = {
    'sigmoid': 'σ'
}


def symbol(name):
    return SYMBOLS.get(name, name)


def draw_net(genome: Genome, view=False, filename=None, show_disabled=False, prune_unused=True, fmt='svg',
             in_labels=None, out_labels=None, title=''):
    if not in_labels:
        in_labels = {}
    elif isinstance(in_labels, list):
        in_labels = dict(enumerate(in_labels))
    if not out_labels:
        out_labels = {}

    inputs = set(ng for ng in genome.node_genes.values() if ng.type == 'INPUT')
    outputs = set(ng for ng in genome.node_genes.values() if ng.type == 'OUTPUT')

    if show_disabled:
        connections = set(genome.conn_genes.values())
    else:
        connections = set(cg for cg in genome.conn_genes.values() if cg.enabled)

    if prune_unused:
        can_reach_from_input = set(ng.ID for ng in inputs)
        can_reach_from_output = set(ng.ID for ng in outputs)

        finished = False
        while not finished:
            finished = True
            for connection in connections:
                a = connection.in_node_id
                b = connection.out_node_id

                if b not in can_reach_from_input and a in can_reach_from_input:
                    can_reach_from_input.add(b)
                    finished = False

                if a not in can_reach_from_output and b in can_reach_from_output:
                    can_reach_from_output.add(a)
                    finished = False

        used_node_ids = can_reach_from_input & can_reach_from_output

        for ng_id in (can_reach_from_output - used_node_ids):
            if genome.node_genes[ng_id].bias:
                used_node_ids.add(ng_id)
    else:
        used_node_ids = set(genome.node_genes.keys())

    dot = graphviz.Digraph(format=fmt, node_attr=REGULAR_NODE_STYLE)

    if title:
        dot.attr('graph', label=title, labelloc='t')

    for ng_id, ng in genome.node_genes.items():
        node_id = str(ng_id)
        label = '{symbol}\nbias: {bias}'.format(
            symbol=symbol(ng.activation_type),
            bias=round(ng.bias, 1)
        )

        if ng.type == 'INPUT':
            in_hidden_node = node_id + '_in'
            sg = graphviz.Digraph()
            sg.node(in_hidden_node, label=in_labels.get(ng_id, str(ng_id)), **INVISIBLE_NODE)
            sg.node(node_id, label=label, **INPUT_NODE_STYLE)
            sg.edge(in_hidden_node, node_id, color=IO_EDGE_COLOR)
            dot.subgraph(sg)
        elif ng.type == 'OUTPUT':
            out_hidden_node = node_id + '_out'
            sg = graphviz.Digraph()
            sg.node(out_hidden_node, label=out_labels.get(ng_id, str(ng_id)), **INVISIBLE_NODE)
            sg.node(node_id, label=label, **OUTPUT_NODE_STYLE)
            sg.edge(node_id, out_hidden_node, color=IO_EDGE_COLOR)
            dot.subgraph(sg)
        elif ng_id in used_node_ids:
            dot.node(node_id, label=label)

    for cg in connections:
        a = cg.in_node_id
        b = cg.out_node_id

        if not ((a in used_node_ids) and (b in used_node_ids)):
            continue

        style = 'solid' if cg.enabled else 'dotted'
        color = 'green' if cg.weight > 0 else 'red'
        width = str(0.1 + abs(cg.weight / 5.0))
        dot.edge(str(a), str(b), label=str(round(cg.weight, 1)), style=style, color=color, width=width)

    dot.render(filename, view=view)

    return dot
