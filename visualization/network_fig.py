from math import atan, cos, sin

from matplotlib import pyplot
from neat.nn import find_feed_forward_layers

# modified from https://github.com/miloharper/visualise-neural-network
# MIT license

vertical_distance_between_layers = 3
horizontal_distance_between_neurons = 2
neuron_radius = 0.5


class Arrow(pyplot.Arrow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, color='#AAAAAA', width=0.2, **kwargs)


class DrawNeuron:
    def __init__(self, x, y, annotation=None):
        self.x = x
        self.y = y
        self.annotation = annotation

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)

        if self.annotation:
            pyplot.gca().annotate(self.annotation, xy=(self.x, self.y), ha='center', va='center', size='xx-large')

    def add_input_arrow(self, annotation=''):
        y = self.y + 2 * neuron_radius
        arrow = Arrow(self.x, y, 0, -neuron_radius)
        pyplot.gca().add_patch(arrow)
        x_offset = 0.15

        if annotation:
            pyplot.gca().annotate(
                annotation, xy=(self.x + x_offset, y), ha='left', va='top', size='x-large', weight='bold'
            )

    def add_output_arrow(self, annotation=''):
        y = self.y - neuron_radius
        arrow = Arrow(self.x, y, 0, -neuron_radius)
        pyplot.gca().add_patch(arrow)
        x_offset = 0.15

        if annotation:
            pyplot.gca().annotate(
                annotation, xy=(self.x + x_offset, y), ha='left', va='top', size='x-large', weight='bold'
            )


class DrawLayer:
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, number_of_layers,
                 annotations=None):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.number_of_layers = number_of_layers

        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, annotations)

    def __intialise_neurons(self, number_of_neurons, annotations):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            if annotations:
                text = annotations[iteration]
            else:
                text = None

            neuron = DrawNeuron(x, self.y, text)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y - vertical_distance_between_layers
        else:
            return vertical_distance_between_layers * self.number_of_layers

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()


class DrawNetwork:
    def __init__(self, number_of_neurons_in_widest_layer, number_of_layers):
        self.layers = []
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.number_of_layers = number_of_layers

    def add_layer(self, number_of_neurons, annotations=None):
        draw_layer = DrawLayer(
            self,
            number_of_neurons,
            number_of_layers=self.number_of_layers,
            number_of_neurons_in_widest_layer=self.number_of_neurons_in_widest_layer,
            annotations=annotations
        )
        self.layers.append(draw_layer)
        return draw_layer

    def connect_neurons(self, neuron1, neuron2, annotation=None, annotation_offset=(0, 0)):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        x0 = neuron2.x - x_adjustment
        x1 = neuron1.x + x_adjustment
        y0 = neuron2.y - y_adjustment
        y1 = neuron1.y + y_adjustment

        arrow = Arrow(x0, y0, (x1 - x0), (y1 - y0))
        pyplot.gca().add_patch(arrow)

        if annotation:
            center_x = x0 + (x1 - x0) / 2
            center_y = y0 + (y1 - y0) / 2
            offset_x, offset_y = annotation_offset
            pyplot.gca().annotate(
                annotation,
                xy=(center_x + offset_x, center_y + offset_y),
                ha='center',
                va='center',
                size='x-large'
            )


SYMBOLS = {
    'sigmoid': r'$\sigma$'
}


def draw_network(network_genome, display='show'):
    input_ids = [ng.ID for ng in network_genome.node_genes.values() if ng.type == 'INPUT']
    output_ids = [ng.ID for ng in network_genome.node_genes.values() if ng.type == 'OUTPUT']
    connection_keys = [(cg.in_node_id, cg.out_node_id) for cg in network_genome.conn_genes.values() if cg.enabled]

    ids_by_layer = [set(input_ids)] + find_feed_forward_layers(input_ids, connection_keys)

    widest_layer = max(map(len, ids_by_layer))
    number_of_layers = len(ids_by_layer)

    network = DrawNetwork(number_of_neurons_in_widest_layer=widest_layer, number_of_layers=number_of_layers)

    id_to_neuron_mapping = {}
    for neuron_ids in ids_by_layer:
        annotations = [SYMBOLS[network_genome.node_genes[id].activation_type] for id in neuron_ids]
        network_layer = network.add_layer(number_of_neurons=len(neuron_ids), annotations=annotations)

        for id, neuron in zip(neuron_ids, network_layer.neurons):
            id_to_neuron_mapping[id] = neuron

    for draw_layer in network.layers:
        draw_layer.draw()

    for connection_key in connection_keys:
        id_a, id_b = connection_key
        w = round(network_genome.conn_genes[connection_key].weight, 1)
        network.connect_neurons(id_to_neuron_mapping[id_b], id_to_neuron_mapping[id_a], annotation=str(w))

    for neuron_id in input_ids:
        id_to_neuron_mapping[neuron_id].add_input_arrow()

    for neuron_id in output_ids:
        id_to_neuron_mapping[neuron_id].add_output_arrow()

    pyplot.axis('scaled')
    pyplot.gca().get_xaxis().set_visible(False)
    pyplot.gca().get_yaxis().set_visible(False)

    for spine in pyplot.gca().spines.values():
        spine.set_visible(False)

    if display == 'show':
        pyplot.show()
    else:
        pyplot.gcf().savefig('out.png', bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    network = DrawNetwork(number_of_neurons_in_widest_layer=2, number_of_layers=3)
    l0 = network.add_layer(2, annotations=[r'$\sigma$', r'$\sigma$'])
    l1 = network.add_layer(2, annotations=[r'$\sin$', r'$\sin$'])
    l2 = network.add_layer(1)

    for layer in network.layers:
        layer.draw()

    pyplot.axis('scaled')
    pyplot.gca().get_xaxis().set_visible(False)
    pyplot.gca().get_yaxis().set_visible(False)

    for spine in pyplot.gca().spines.values():
        spine.set_visible(False)

    network.connect_neurons(l1.neurons[0], l0.neurons[0], '0.5')
    network.connect_neurons(l1.neurons[0], l0.neurons[1], 'a', annotation_offset=(-0.5, 0.5))
    network.connect_neurons(l1.neurons[1], l0.neurons[0], 'b', annotation_offset=(0.5, 0.5))
    network.connect_neurons(l1.neurons[1], l0.neurons[1], '0.5')
    network.connect_neurons(l2.neurons[0], l1.neurons[0], '0.5')
    network.connect_neurons(l2.neurons[0], l1.neurons[1], '0.5')

    l0.neurons[0].add_input_arrow('x')
    l0.neurons[1].add_input_arrow('y')
    l2.neurons[0].add_output_arrow()

    pyplot.show()
    # pyplot.gcf().savefig('out.png', bbox_inches='tight', pad_inches=0.1)
