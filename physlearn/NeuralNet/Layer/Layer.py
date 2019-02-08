import numpy
import tensorflow as tf


class Layer:
    weight_dim = 0
    bias_vector = None
    bias_dim = 0
    dim = 0
    weight_matrix = None

    def __init__(self, shape=None, activation_func=None, index=None):
        self.weight_matrix_shape = shape
        self.activation_func = activation_func
        self.index = index
        bias_np = numpy.random.uniform(-1, 1, (self.bias_dim, 1))
        self.bias_vector = tf.Variable(bias_np, dtype=tf.double)

    def set_params(self, weight_matrix, weights_matrix_shape, weight_dim, bias_vector, bias_dim, activation_func, dim):
        pass

    def create_layer_from_params(self, unroll_vector=None):
        pass

    def roll_matrix(self, layer_unroll_vector):
        weight_unroll_vector = layer_unroll_vector[:self.weight_dim]
        bias_unroll_vector = layer_unroll_vector[self.weight_dim:]

        bias_vector = tf.reshape(bias_unroll_vector, (self.bias_dim, 1))
        self.bias_vector = bias_vector

        return weight_unroll_vector

    def return_layer_dim(self):
        return self.dim

    def __mul__(self, other):
        return self.weight_matrix * other + self.bias_vector

    @staticmethod
    def return_class_name():
        pass
