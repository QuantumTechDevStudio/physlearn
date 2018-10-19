import tensorflow as tf

from physlearn.NeuralNet.MatrixA import MatrixA


class MatrixGen(MatrixA):

    def __mul__(self, x):
        return tf.matmul(self.matrix, x)
