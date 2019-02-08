import tensorflow as tf

from physlearn.NeuralNet.Matrixes.MatrixA import MatrixA


class MatrixGen(MatrixA):

    def __mul__(self, x):
        return tf.matmul(self.matrix, x)

    def roll_matrix(self, unroll_vector):
        matrix = tf.reshape(unroll_vector, self.shape)
        self.matrix = matrix
