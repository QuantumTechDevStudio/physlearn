from physlearn.NeuralNet.Layer.LayerBlocks import LayerBlocks
from physlearn.NeuralNet.Layer.LayerFC import LayerFC


class LayerCreator:
    id_list = []
    layer_types = []

    def create_layer(self, layer_num, prev_layer_units=None, next_layer_units=None, activation_func=None,
                     layer_unroll_vector=None):
        if layer_num in self.id_list:
            layer_type = self.layer_types[self.id_list.index(layer_num)]
            if layer_type == 0:
                layer = LayerFC(layer_id=layer_num, layer_unroll_vector=layer_unroll_vector)
            else:
                layer = LayerBlocks(layer_id=layer_num, layer_unroll_vector=layer_unroll_vector)

        else:
            if type(next_layer_units) != list:
                self.id_list.append(layer_num)
                self.layer_types.append(0)

                if type(prev_layer_units) == list:
                    layer = self.__create_fc_layer(layer_num, sum(prev_layer_units), next_layer_units, activation_func)
                else:
                    layer = self.__create_fc_layer(layer_num, prev_layer_units, next_layer_units, activation_func)

            else:
                self.id_list.append(layer_num)
                self.layer_types.append(1)
                layer = self.__create_block_layers(layer_num, prev_layer_units, next_layer_units, activation_func)

        return layer

    @staticmethod
    def __create_fc_layer(layer_num, prev_layer_units, next_layer_units, activation_func):
        shape = (next_layer_units, prev_layer_units)
        return LayerFC(layer_num, shape, activation_func)

    @staticmethod
    def __create_block_layers(layer_num, prev_layer_units, next_layer_units, activation_func):
        shapes = []
        for index, _ in enumerate(prev_layer_units):
            shapes.append((next_layer_units[index], prev_layer_units[index]))
        return LayerBlocks(layer_num, shapes, activation_func)
