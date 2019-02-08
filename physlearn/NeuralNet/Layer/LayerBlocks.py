import numpy
import tensorflow as tf

from physlearn.NeuralNet.Layer.Layer import Layer
from physlearn.NeuralNet.Matrixes.MatrixBlocks import MatrixBlocks


class LayerBlocks(Layer):
    def __init__(self, shape=None, activation_func=None, index=None):
        self.weight_dim = 0
        self.bias_dim = 0
        self.break_points = []
        if not (shape is None):
            for cur_shape in shape:
                self.bias_dim += cur_shape[0]
                self.weight_dim += cur_shape[0] * cur_shape[1]
            prev_break_point = 0
            for cur_shape in shape:
                cur_break_point = prev_break_point + (cur_shape[0] * cur_shape[1])
                self.break_points.append(cur_break_point)
                prev_break_point = cur_break_point

            self.dim = self.weight_dim + self.bias_dim
            self.weight_matrix = MatrixBlocks([tf.Variable(expected_shape=shape[i], dtype=tf.double)
                                               for i in range(len(shape))], shape)
        super().__init__(shape, activation_func, index)

    def set_params(self, weight_matrix, weights_matrix_shape, weight_dim, bias_vector, bias_dim, activation_func, dim):
        self.weight_matrix = weight_matrix
        self.weight_matrix_shape = weights_matrix_shape
        self.weight_dim = weight_dim
        self.bias_vector = bias_vector
        self.bias_dim = bias_dim
        self.activation_func = activation_func
        self.dim = dim

    def create_layer_from_params(self, unroll_vector=None):
        for cur_shape in self.weight_matrix_shape:
            self.bias_dim += cur_shape[0]
            self.weight_dim += cur_shape[0] * cur_shape[1]
        self.dim = self.weight_dim + self.bias_dim
        if unroll_vector is None:
            self.weight_matrix = MatrixBlocks([tf.Variable(expected_shape=self.weight_matrix_shape[i], dtype=tf.double)
                                               for i in range(len(self.weight_matrix_shape))], self.weight_matrix_shape)
        else:
            weight_unroll_vector = self.roll_matrix(unroll_vector)
            matrix_list = []
            for index in range(self.weight_matrix_shape):
                left_break = self.break_points[index]
                right_break = self.break_points[index + 1]
                cur_unroll_vector = weight_unroll_vector[left_break:right_break]
                matrix_list.append(tf.reshape(cur_unroll_vector, self.weight_matrix_shape[index]))
            self.weight_matrix = MatrixBlocks(matrix_list, self.weight_matrix_shape)

    @staticmethod
    def return_class_name():
        return 'LayerBlocks'
