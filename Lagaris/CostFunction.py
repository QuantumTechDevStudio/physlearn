# Необходмые команды импорта.
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import math_util


class CostFunction:

    def __init__(self, net, ground_method = 'no_gaus'):
        self.net = None
        self.net_output = None
        self.func_sum = None
        self.beta = None
        self.trial_func = None
        self.beta_loc = None
        self.session = None
        self.assign_beta = None
        self.ground_cond_method = None
        self.noninvariance_factor = None
        self.cost_func = None
        self.approximation_grid = None
        self.linearity_grid = None
        self.M = None
        self.a = None
        self.b = None
        self.m = None
        self.ground_cond_method = ground_method
        self.set_net(net)

            
    def set_net(self, net):
        self.net = net
        self.session = net.return_session()
        self.beta = tf.get_variable("beta", shape=(), dtype=tf.float64, initializer=tf.ones_initializer)
        self.session.run(self.beta.initializer)
        self.net_output = net.return_graph()
        self.beta_loc = tf.placeholder(tf.float64, shape = ())
        self.assign_beta = self.beta.assign(self.beta_loc)
        if self.ground_cond_method == 'gaus':
            self.trial_func = tf.exp(-tf.abs(self.beta)*tf.square(net.x))*self.net_output
        else:
            self.trial_func = self.net_output
        self.func_sum = tf.reduce_sum(input_tensor = self.trial_func, axis=0)
        

    def define_approximation_grid(self, a, b, m):
        self.m = m
        self.a = a
        self.b = b
        self.approximation_grid = np.linspace(a, b, m, endpoint=True).reshape(1, m)

        
    def define_linearity_grid(self, M):
        self.M = M
        self.linearity_grid = math_util.calc_hermite_grid_xi(M).reshape(1, M)

        
    def calc_linearity_factor(self):
        colloc_matrix = self.net.calc(self.trial_func, {self.net.x: self.linearity_grid})
        diag_colloc_matrix = np.eye(self.M)
        diag_colloc_matrix.flat[:: self.M + 1] += -1 + colloc_matrix.diagonal()
        normal_colloc_matrix = np.matmul(LA.inv(diag_colloc_matrix), colloc_matrix)
        np.fill_diagonal(normal_colloc_matrix, 0)
        linear_factor = math_util.norm(normal_colloc_matrix)
        return linear_factor

    
    def define_noninvariance_factor(self):
        first_deriative = tf.gradients(self.trial_func, self.net.x)[0]
        # Hamiltonian image definition
        net_images = (-(tf.gradients(first_deriative, self.net.x)[0]) + tf.multiply(tf.square(self.net.x), self.trial_func))
        net_images_sum = tf.reduce_sum(input_tensor=net_images, axis=0)
        A = tf.transpose(self.trial_func)
        A_T = self.trial_func
        y = net_images_sum
        y = tf.expand_dims(y, -1)
        omega = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(A_T, A)), A_T), y)
        regression_fit = tf.matmul(tf.transpose(self.trial_func), omega)
        self.noninvariance_factor = (1 / self.m) * tf.reduce_sum(tf.square(tf.expand_dims(net_images_sum, -1) - regression_fit))

                      
    def calc_ground_cond_factor(self):
        ground_cond_factor = np.sum(np.square(self.net.run(np.linspace(-8, 8, 2, endpoint=True).reshape(1, 2))))
        ground_cond_factor += 2 * np.sum(np.square(self.net.run(np.linspace(-9, 9, 2, endpoint=True).reshape(1, 2))))
        ground_cond_factor += 3 * np.sum(np.square(self.net.run(np.linspace(-10, 10, 2, endpoint=True).reshape(1, 2))))
        return ground_cond_factor

                      
    def J(self, params):
        self.net.roll_matrixes(params)          
        j = self.net.calc(self.noninvariance_factor, {self.net.x: self.approximation_grid})
        j += self.calc_linearity_factor()
        j += self.calc_ground_cond_factor()
        return j

    def J_gaus_ground(self, params):
        N = params.size
        self.net.roll_matrixes(params[:N-1])
        self.session.run(self.assign_beta, {self.beta_loc : params[N-1]})
        j = self.net.calc(self.noninvariance_factor, {self.net.x: self.approximation_grid})
        j += self.calc_linearity_factor()
        return j                      
                      
                      
    def compile(self):
        self.define_noninvariance_factor()                      
        if self.ground_cond_method == 'gaus':
            self.cost_func = self.J_gaus_ground
        else:
            self.cost_func = self.J

                      
    def get_cost_func(self):                  
        return self.cost_func
