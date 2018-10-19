import tensorflow as tf

from physlearn.NeuralNet.NeuralNetAbstract import NeuralNetAbstract


class NeuralNet(NeuralNetAbstract):

    def add_sub_nets(self, sub_nets):
        self.design.append((sub_nets, 1))

    def __add_fl_layer(self, index):
        current_layer_units = self.design[index][0]
        next_layer_units = self.design[index + 1][0]
        size = current_layer_units * next_layer_units  # Количество элементов матрицы
        weight_breaker = size + self.unroll_breaks[index][1]  # Индекс конца матрицы весов в unroll векторе -
        # ее размер, плюс сдвиг, связанный с предыдущими матрицами
        bias_breaker = weight_breaker + next_layer_units  # Аналогично
        self.unroll_breaks.append((weight_breaker, bias_breaker))
        tf_weight_matrix = tf.placeholder(tf.double)
        tf_bias_vector = tf.placeholder(tf.double)
        self.size_list.append(((next_layer_units, current_layer_units), (next_layer_units, 1)))
        return tf_weight_matrix, tf_bias_vector

    def create_tf_matrixes(self):
        tf_matrixes = []
        for index in range(len(self.design) - 1):
            if self.design[index][-1] == 0:
                tf_matrixes.append(self.__add_fl_layer(index))
            else:
                sub_nets = self.design[index][0]
                sub_sizes = []
                for net in sub_nets:
                    sub_sizes.append(net.return_sizes())
                total_layers = sub_nets[0].amount_of_layers
                layers = []
                for i in range(total_layers):
                    cur_layer = []
                    for cur_net in sub_sizes:
                        cur_layer.append(cur_net[i])
                    layers.append(cur_layer)
