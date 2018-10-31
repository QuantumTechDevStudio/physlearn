import tensorflow as tf


class Layer:
    activation_func = None
    weight_matrix = None
    bias_vector = None
    weight_matrix_shape = None
    bias_dim = 0
    weight_dim = 0
    dim = 0

    def __init__(self, shape, activation_func):
        self.bias_vector = tf.placeholder(tf.double)
        self.weight_matrix_shape = shape
        self.activation_func = activation_func

    def roll_matrix(self, layer_unroll_vector):
        weight_unroll_vector = layer_unroll_vector[:self.weight_dim]
        bias_unroll_vector = layer_unroll_vector[self.weight_dim:]
        assign_list = self.weight_matrix.return_assign_list(weight_unroll_vector)
        bias_assign_vector = bias_unroll_vector.reshape(self.bias_dim, 1)
        assign_list.append((self.bias_vector, bias_assign_vector))
        return assign_list

    def return_layer_dim(self):
        return self.dim

    def __mul__(self, other):
        return self.weight_matrix * other + self.bias_vector
