import tensorflow as tf

from physlearn.NeuralNet.Layer.Layer import Layer
from physlearn.NeuralNet.Matrixes.MatrixGen import MatrixGen


class LayerFC(Layer):
    def __init__(self, shape, activation_func):
        self.weight_matrix = MatrixGen(tf.placeholder(tf.double), shape)
        self.bias_dim = shape[0]
        super().__init__(shape, activation_func)
