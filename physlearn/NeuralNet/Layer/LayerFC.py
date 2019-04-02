import numpy
import tensorflow as tf

from physlearn.NeuralNet.Layer.Layer import Layer
from physlearn.NeuralNet.Matrixes.MatrixGen import MatrixGen


class LayerFC(Layer):

    def create_layer_params(self, shape, activation_func):
        params = {'weight_matrix_shape': shape}
        weight_dim = params['weight_dim'] = shape[0] * shape[1]
        bias_dim = params['bias_dim'] = shape[0]
        params['dim'] = weight_dim + bias_dim
        params['activation_func'] = activation_func
        return params

    def create_matrixes(self, shape):
        weight_np = numpy.random.uniform(-1, 1, shape)
        weight_matrix = MatrixGen(tf.Variable(weight_np, dtype=tf.double), shape)
        return weight_matrix

    def create_matrix_from_unroll_vector(self, unroll_vector, shape):
        weight_matrix = MatrixGen(unroll_vector, shape, matrix_type=1)
        return weight_matrix
