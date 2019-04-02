import numpy
import tensorflow as tf

from tqdm import tqdm


class NelderMeadTF:

    def __init__(self):
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

        self.user_x = None

    def variant1(self):
        self.x_stretch = (self.gamma * self.x_reflected) + ((1 - self.gamma) * self.x_center)
        self.y_stretch = self.func(self.x_stretch)
        result = tf.cond(self.y_stretch < self.y_points[self.l_index], self.variant1_1, self.variant1_2)
        return result

    def variant1_1(self):
        return self.x_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.x_stretch, (1, self.dim)))

    def variant1_2(self):
        return self.x_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.x_reflected, (1, self.dim)))

    def variant2(self):
        result = tf.cond(self.y_reflected < self.y_points[self.g_index], self.variant2_1, self.variant3)
        return result

    def variant2_1(self):
        return self.x_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.x_reflected, (1, self.dim)))

    def variant3(self):
        res = tf.cond(self.y_reflected < self.y_points[self.h_index], self.replace_1, self.replace_2)
        self.new_x_points = self.x_points
        y_h = res[1]
        x_h = self.new_x_points[self.h_index]

        self.x_compress = (self.beta * x_h) + ((1 - self.beta) * self.x_center)
        self.y_compress = self.func(self.x_compress)

        result = tf.cond(self.y_compress < y_h, self.variant3_1, self.compress_simplex)
        return result

    def compress_simplex(self):
        return self.x_points.assign(0.5 * (self.new_x_points + self.x_l))

    def variant3_1(self):
        return self.x_points.scatter_nd_update(self.h_index_replace, tf.reshape(self.x_compress, (1, self.dim)))

    def replace_1(self):
        return self.x_points.scatter_nd_update(self.h_index_replace,
                                               tf.reshape(self.x_reflected, (1, self.dim))), self.y_reflected

    def replace_2(self):
        return self.x_points, self.y_points[self.h_index]

    def set_x(self, x):
        self.user_x = x

    def optimize(self, func, dim, end_cond, sess, placeholder=None):
        self.func = func
        self.dim = dim
        self.x_points = tf.Variable(numpy.random.uniform(-10, 10, (dim + 1, dim)), dtype=tf.float64)
        sess.run(tf.global_variables_initializer())
        self.y_points = tf.map_fn(func, self.x_points)

        values, indices = tf.math.top_k(self.y_points, k=2)
        # [h_index, g_index] = indices
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
        cost_list = []
        for _ in tqdm(range(end_cond)):
            _, cost, l_index_np = sess.run([result, self.y_points, self.l_index], {placeholder: self.user_x})
            cost_list.append(cost[l_index_np])

        x_points_np, l_index_np = sess.run([self.x_points, self.l_index], {placeholder: self.user_x})
        return x_points_np[l_index_np], cost_list
