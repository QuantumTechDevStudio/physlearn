import tensorflow as tf

from physlearn.NeuralNet.Layer.Layer import Layer
from physlearn.NeuralNet.Matrixes.MatrixBlocks import MatrixBlocks


class LayerBlocks(Layer):
    def __init__(self, shape, activation_func):
        self.weight_dim = 0
        self.bias_dim = 0
        for cur_shape in shape:
            self.bias_dim += cur_shape[0]
            self.weight_dim += cur_shape[0] * cur_shape[1]
        self.dim = self.weight_dim + self.bias_dim
        self.weight_matrix = MatrixBlocks([tf.placeholder(tf.double) for _ in range(len(shape))], shape)
        super().__init__(shape, activation_func)

