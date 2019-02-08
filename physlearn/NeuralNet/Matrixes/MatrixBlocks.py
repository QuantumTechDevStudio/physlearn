import tensorflow as tf

from physlearn.NeuralNet.Matrixes.MatrixA import MatrixA


class MatrixBlocks(MatrixA):

    def __init__(self, matrix, shape):
        self.break_points = [0]
        #self.matrix = matrix

        super().__init__(matrix, shape)

    def __mul__(self, x):
        prev_break_point = 0
        res_list = []
        for index, matrix in enumerate(self.matrix):
            cur_break_point = prev_break_point + self.shape[index][1]
            res_list.append(tf.matmul(matrix, x[prev_break_point:cur_break_point]))
            prev_break_point = cur_break_point
        res = tf.concat(res_list, 0)
        return res

    def roll_matrix(self, unroll_vector):
        for index, _ in enumerate(self.matrix):
            left_break = self.break_points[index]
            right_break = self.break_points[index + 1]
            cur_unroll_vector = unroll_vector[left_break:right_break]
            self.matrix[index] = tf.reshape(cur_unroll_vector, self.shape[index])
