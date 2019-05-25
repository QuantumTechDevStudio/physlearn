import numpy
import tensorflow as tf

from physlearn.NeuralNet.Layer.Layer import Layer
from physlearn.NeuralNet.Matrixes.MatrixBlocks import MatrixBlocks


class LayerBlocks(Layer):

    def create_layer_params(self, shape, activation_func):
        params = {'weight_matrix_shape': shape}

        weight_dim = 0
        bias_dim = 0

        for cur_shape in shape:
            bias_dim += cur_shape[0]
            weight_dim += cur_shape[0] * cur_shape[1]

        dim = weight_dim + bias_dim

        params['weight_dim'] = weight_dim
        params['bias_dim'] = bias_dim
        params['dim'] = dim
        params['activation_func'] = activation_func
        return params

    def create_matrixes(self, shape):
        weight_matrix = MatrixBlocks([tf.Variable(numpy.random.uniform(-1, 1), dtype=tf.double)
                                      for _ in range(len(shape))], shape)
        return weight_matrix

    def create_matrix_from_unroll_vector(self, unroll_vector, shape):
        weight_matrix = MatrixBlocks(unroll_vector, shape, matrix_type=1)
        return weight_matrix
