import tensorflow as tf

from physlearn.NeuralNet.Layer.Layer import Layer
from physlearn.NeuralNet.Matrixes.MatrixBlocks import MatrixBlocks


class LayerBlocks(Layer):
    def __init__(self, shape, activation_func):
        self.bias_dim = sum(list(map(lambda s: s[0], shape)))
        self.weight_matrix = MatrixBlocks([tf.placeholder(tf.double) for _ in range(len(shape))], shape)
        super().__init__(shape, activation_func)
