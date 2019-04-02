import numpy
import tensorflow as tf


class Layer:
    params_list = []
    id_list = []

    def __init__(self, layer_id, shape=None, activation_func=None, layer_unroll_vector=None):
        self.bias_assign = None
        self.bias_placeholder = None
        if layer_id in self.id_list:
            params = self.params_list[self.id_list.index(layer_id)]
        else:
            self.id_list.append(layer_id)
            params = self.create_layer_params(shape, activation_func)
            self.params_list.append(params)

        self.__set_params(params)

        if layer_unroll_vector is None:
            self.weight_matrix = self.create_matrixes(self.weight_matrix_shape)
            bias_np = numpy.random.uniform(-1, 1, (self.bias_dim, 1))
            self.bias_vector = tf.Variable(bias_np, dtype=tf.double)

        else:
            weight_unroll_vector = self.roll_matrix(layer_unroll_vector)
            self.weight_matrix = self.create_matrix_from_unroll_vector(weight_unroll_vector, self.weight_matrix_shape)

    def __set_params(self, params):
        self.weight_matrix_shape = params['weight_matrix_shape']
        self.weight_dim = params['weight_dim']
        self.bias_dim = params['bias_dim']
        self.dim = params['dim']
        self.activation_func = params['activation_func']

    def create_layer_params(self, shape, activation_func):
        pass

    def create_matrixes(self, shape):
        pass

    def create_matrix_from_unroll_vector(self, unroll_vector, shape):
        pass

    def roll_matrix(self, layer_unroll_vector):
        weight_unroll_vector = layer_unroll_vector[:self.weight_dim]
        bias_unroll_vector = layer_unroll_vector[self.weight_dim:]

        bias_vector = tf.reshape(bias_unroll_vector, (self.bias_dim, 1))
        self.bias_vector = bias_vector

        return weight_unroll_vector

    def create_assigns(self):
        self.bias_placeholder = tf.placeholder(dtype=tf.double)
        self.bias_assign = self.bias_vector.assign(self.bias_placeholder)
        matrix_assign = self.weight_matrix.create_assigns()
        assign_list = [self.bias_assign]
        assign_list.extend(matrix_assign)
        return assign_list

    def assign_matrixes(self, layer_unroll_vector):
        weight_unroll_vector = layer_unroll_vector[:self.weight_dim]
        bias_unroll_vector = layer_unroll_vector[self.weight_dim:]
        bias_assign_vector = numpy.reshape(bias_unroll_vector, (self.bias_dim, 1))
        placeholder_dict = {self.bias_placeholder: bias_assign_vector}
        matrixes_placeholder_dict = self.weight_matrix.assign_matrixes(weight_unroll_vector)
        placeholder_dict.update(matrixes_placeholder_dict)
        return placeholder_dict

    def return_layer_dim(self):
        return self.dim

    def calc_output(self, input_vector):
        return self.activation_func(self.weight_matrix * input_vector + self.bias_vector)
