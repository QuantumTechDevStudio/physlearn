import tensorflow as tf
import numpy

from physlearn.NeuralNet.Matrixes.MatrixA import MatrixA


class MatrixGen(MatrixA):
    def __mul__(self, x):
        return tf.matmul(self.matrix, x)

    def roll_matrix(self, unroll_vector):
        matrix = tf.reshape(unroll_vector, self.shape)
        self.matrix = matrix

    def create_assigns(self):
        self.matrix_placeholder = tf.placeholder(dtype=tf.double)
        self.matrix_assign = self.matrix.assign(self.matrix_placeholder)
        return [self.matrix_assign]

    def assign_matrixes(self, unroll_vector):
        matrix = numpy.reshape(unroll_vector, self.shape)
        placeholder_dict = {self.matrix_placeholder: matrix}
        return placeholder_dict
