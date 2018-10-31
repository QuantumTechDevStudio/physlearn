import tensorflow as tf

from physlearn.NeuralNet.Matrixes.MatrixA import MatrixA


class MatrixBlocks(MatrixA):

    def __init__(self, matrix, shape):
        self.break_points = [0]
        self.matrix = None
        self.shape = None
        prev_break_point = 0
        for cur_shape in shape:
            cur_break_point = prev_break_point + (cur_shape[0] * cur_shape[1])
            self.break_points.append(cur_break_point)
            prev_break_point = cur_break_point
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

    def return_assign_list(self, unroll_vector):
        assign_list = []
        for i, matrix in enumerate(self.matrix):
            assign_matrix = unroll_vector[self.break_points[i]:self.break_points[i + 1]].reshape(self.shape[i])
            assign_list.append((matrix, assign_matrix))
        return assign_list
