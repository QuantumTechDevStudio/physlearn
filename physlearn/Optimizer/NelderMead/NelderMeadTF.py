import time

import numpy
import tensorflow as tf

from physlearn.Optimizer.OptimizeResult import OptimizeResult
from physlearn.Optimizer.OptimizerAbstract import OptimizerAbstract


class NelderMeadTF(OptimizerAbstract):

    def __init__(self, min_element=-1, max_element=1):
        super().__init__(min_element, max_element)
        self.x_stretch = None
        self.y_stretch = None

        self.x_compress = None
        self.y_compress = None
        self.new_x_points = None

        self.alpha = 1
        self.beta = 0.5
        self.gamma = 2

        self.x_points = None
        self.y_points = None
        self.x_reflected = None
        self.y_reflected = None

        self.func = None

        self.h_index = self.g_index = self.l_index = None
        self.h_index_replace = None

        self.x_h = self.x_g = self.x_l = None

        self.x_center = None

        self.dim = 0

        self.x = None
        self.placeholder = None
        self.sess = None

    def parse_params(self, params_dict):
        self.alpha = params_dict['alpha']
        self.beta = params_dict['beta']
        self.gamma = params_dict['gamma']
        self.sess = params_dict['sess']
        self.placeholder = params_dict['placeholder']
        self.x = params_dict['x']

    def variant1(self):
        self.x_stretch = (self.gamma * self.x_reflected) + ((1 - self.gamma) * self.x_center)
        self.y_stretch = self.func(self.x_stretch)
        result = tf.cond(self.y_stretch < self.y_points[self.l_index], self.variant1_1, self.variant1_2)
        return result

    def variant1_1(self):
        x_update = self.x_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.x_stretch, (1, self.dim)))
        y_update = self.y_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.y_stretch, (1,)))
        return x_update, y_update

    def variant1_2(self):
        x_update = self.x_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.x_reflected, (1, self.dim)))
        y_update = self.y_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.y_reflected, (1,)))
        return x_update, y_update

    def variant2(self):
        result = tf.cond(self.y_reflected < self.y_points[self.g_index], self.variant2_1, self.variant3)
        return result

    def variant2_1(self):
        x_update = self.x_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.x_reflected, (1, self.dim)))
        y_update = self.y_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.y_reflected, (1,)))
        return x_update, y_update

    def variant3(self):
        res = tf.cond(self.y_reflected < self.y_points[self.h_index], self.replace_1, self.replace_2)
        self.new_x_points = res[0]
        y_h = res[1]
        x_h = self.new_x_points[self.h_index]

        self.x_compress = (self.beta * x_h) + ((1 - self.beta) * self.x_center)
        self.y_compress = self.func(self.x_compress)

        result = tf.cond(self.y_compress < y_h, self.variant3_1, self.compress_simplex)
        return result

    def compress_simplex(self):
        x_update = self.x_points.assign(0.5 * (self.new_x_points + self.x_l))
        new_y = lambda: tf.map_fn(self.func, self.x_points.read_value(), dtype=tf.float64)
        self.y_points = tf.Variable(initial_value=new_y)
        self.sess.run(self.y_points.initializer, self.placeholder_dict)
        # print(new_y)
        #y_update = self.y_points.assign(new_y)
        return x_update, self.y_points

    def variant3_1(self):
        x_update = self.x_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.x_compress, (1, self.dim)))
        y_update = self.y_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.y_compress, (1,)))
        return x_update, y_update

    def replace_1(self):
        return self.x_points.scatter_nd_update(self.h_index_replace,
                                               tf.reshape(self.x_reflected, (1, self.dim))), self.y_reflected

    def replace_2(self):
        return self.x_points, self.y_points[self.h_index]

    def set_x(self, x):
        self.x = x

    def set_sess(self, sess):
        self.sess = sess

    def set_placeholder(self, placeholder):
        self.placeholder = placeholder

    def optimize(self, func, dim, end_cond=None, max_time=None, min_func_value=None):
        self.func = func
        self.dim = dim

        if (self.placeholder is None) and (self.x is None):
            self.placeholder_dict = {}
        else:
            self.placeholder_dict = {self.placeholder: self.x}

        self.x_points = tf.Variable(numpy.random.uniform(self.min_element, self.max_element, (dim + 1, dim)),
                                    dtype=tf.float64)

        self.indices = tf.constant(numpy.array([i for i in range(dim + 1)]))
        self.sess.run(tf.global_variables_initializer())
        self.y_points = tf.Variable(tf.map_fn(self.func, self.x_points), dtype=tf.float64)

        self.sess.run(self.y_points.initializer, self.placeholder_dict)

        values, indices = tf.math.top_k(self.y_points, k=2)
        self.h_index = indices[0]
        self.g_index = indices[1]
        self.h_index_replace = tf.reshape(self.h_index, (1, 1))
        self.x_h = self.x_points[self.h_index]
        self.x_g = self.x_points[self.g_index]
        self.l_index = tf.argmin(self.y_points)
        self.x_l = self.x_points[self.l_index]
        self.x_center = (tf.reduce_sum(self.x_points, 0) - self.x_h) / dim
        self.x_reflected = ((1 + self.alpha) * self.x_center) - (self.alpha * self.x_h)
        self.y_reflected = func(self.x_reflected)

        result = tf.cond(self.y_reflected < self.y_points[self.l_index], self.variant1, self.variant2)
        run_list = [*result, self.l_index]
        print(run_list)
        cost_list = []
        i = 0
        start_time = time.time()
        reason_of_break = 'Maximum iterations reached'
        exit_code = -1
        is_converged = False
        while (end_cond is None) or (i <= end_cond):
            # print(self.sess.run(run_list[1], placeholder_dict))
            _, cost, l_index_np = self.sess.run(run_list, self.placeholder_dict)
            min_cost = min(cost)
            cost_list.append(min_cost)
            i += 1
            if (not (min_func_value is None)) and (min_cost < min_func_value):
                reason_of_break = 'Minimum function value reached'
                exit_code = 0
                is_converged = True
                break
            cur_time = time.time() - start_time
            if (not (max_time is None)) and (cur_time > max_time):
                reason_of_break = 'Maximum time reached'
                exit_code = -2
                break

        end_time = time.time()
        total_time = end_time - start_time
        i -= 1
        funcs = self.sess.run(self.y_points, self.placeholder_dict)
        min_index = numpy.argmin(funcs)
        result = OptimizeResult(is_converged, i, total_time, cost_list, exit_code, reason_of_break,
                                self.x_points[min_index])
        return result
