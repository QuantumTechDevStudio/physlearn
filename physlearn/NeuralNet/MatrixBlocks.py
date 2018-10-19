import tensorflow as tf

from physlearn.NeuralNet.MatrixA import MatrixA


class MatrixBlocks(MatrixA):
    def __mul__(self, x):
        prev_break_point = 0
        res_list = []
        for index, matrix in enumerate(self.matrix):
            cur_break_point = prev_break_point + self.shape[index][1]
            res_list.append(tf.matmul(matrix, x[prev_break_point:cur_break_point]))
            prev_break_point = cur_break_point
        res = tf.concat(res_list, 0)
        return res
