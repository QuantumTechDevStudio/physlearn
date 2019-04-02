import tensorflow as tf
import numpy

from physlearn.NeuralNet.Matrixes.MatrixA import MatrixA


class MatrixBlocks(MatrixA):

    def __init__(self, matrix, shape, matrix_type=0):
        self.break_points = [0]
        self.vector_break_points = [0]
        for cur_shape in shape:
            cur_break_point = self.break_points[-1] + (cur_shape[0] * cur_shape[1])
            cur_vector_break_point = self.vector_break_points[-1] * cur_shape[1]
            self.break_points.append(cur_break_point)
            self.vector_break_points.append(cur_vector_break_point)
        super().__init__(matrix, shape, matrix_type)

    def __mul__(self, x):
        res_list = []
        for index, matrix in enumerate(self.matrix):
            left_break_point = self.vector_break_points[index]
            right_break_point = self.vector_break_points[index + 1]
            res_list.append(tf.matmul(matrix, x[left_break_point:right_break_point]))
        res = tf.concat(res_list, 0)
        return res

    def roll_matrix(self, unroll_vector):
        for index, _ in enumerate(self.matrix):
            left_break = self.break_points[index]
            right_break = self.break_points[index + 1]
            cur_unroll_vector = unroll_vector[left_break:right_break]
            self.matrix[index] = tf.reshape(cur_unroll_vector, self.shape[index])

    def create_assigns(self):
        self.matrix_placeholder = []
        self.matrix_assign = []
        for matrix in self.matrix:
            cur_placeholder = tf.placeholder(dtype=tf.double)
            cur_assign = matrix.assign(cur_placeholder)
            self.matrix_placeholder.append(cur_placeholder)
            self.matrix_assign.append(cur_assign)
        return self.matrix_assign

    def assign_matrixes(self, unroll_vector):
        matrixes = []
        for index, _ in enumerate(self.matrix):
            left_break = self.break_points[index]
            right_break = self.break_points[index + 1]
            cur_unroll_vector = unroll_vector[left_break:right_break]
            matrixes.append(numpy.reshape(cur_unroll_vector, self.shape[index]))
        placeholder_dict = {self.matrix_placeholder[i]: matrixes[i] for i in range(len(matrixes))}
        return placeholder_dict
