import tensorflow as tf

from physlearn.NeuralNet.Matrixes.MatrixA import MatrixA


class MatrixGen(MatrixA):

    def __mul__(self, x):
        return tf.matmul(self.matrix, x)

    def return_assign_list(self, unroll_vector):
        roll_matrix = unroll_vector.reshape(self.shape)
        return [(self.matrix, roll_matrix)]
