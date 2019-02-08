import numpy
import tensorflow as tf

from physlearn.NeuralNet.Layer.Layer import Layer
from physlearn.NeuralNet.Matrixes.MatrixGen import MatrixGen


class LayerFC(Layer):
    def __init__(self, shape=None, activation_func=None, index=None):
        self.weight_matrix = None
        self.bias_dim = 0
        self.dim = 0
        if not (shape is None):
            weight_np = numpy.random.uniform(-1, 1, shape)
            self.weight_matrix = MatrixGen(tf.Variable(weight_np, dtype=tf.double), shape)

            self.weight_dim = shape[0] * shape[1]
            self.bias_dim = shape[0]
            self.dim = self.weight_dim + self.bias_dim

        super().__init__(shape, activation_func, index)

    def set_params(self, weight_matrix, weights_matrix_shape, weight_dim, bias_vector, bias_dim, activation_func, dim):
        self.weight_matrix = weight_matrix
        self.weight_matrix_shape = weights_matrix_shape
        self.weight_dim = weight_dim
        self.bias_vector = bias_vector
        self.bias_dim = bias_dim
        self.activation_func = activation_func
        self.dim = dim

    def create_layer_from_params(self, unroll_vector=None):
        if unroll_vector is None:
            self.weight_matrix = MatrixGen(tf.zeros(shape=self.weight_matrix_shape, dtype=tf.double),
                                           self.weight_matrix_shape)
        else:
            weight_unroll_vector = self.roll_matrix(unroll_vector)
            weight_matrix = tf.reshape(weight_unroll_vector, self.weight_matrix_shape)
            self.weight_matrix = MatrixGen(weight_matrix, self.weight_matrix_shape)

        self.weight_dim = self.weight_matrix_shape[0] * self.weight_matrix_shape[1]
        self.bias_dim = self.weight_matrix_shape[0]
        self.dim = self.weight_dim + self.bias_dim

    @staticmethod
    def return_class_name():
        return 'LayerFC'
