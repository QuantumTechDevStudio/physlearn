import time

import numpy
import tensorflow as tf

from physlearn.Optimizer.OptimizeResult import OptimizeResult
from physlearn.Optimizer.OptimizerAbstract import OptimizerAbstract


class DifferentialEvolutionTF(OptimizerAbstract):
    def __init__(self, min_element=-1, max_element=1):
        super().__init__(min_element, max_element)
        self.f = 0.5
        self.p = 0.9
        self.dim = -1
        self.number_of_individuals = -1
        self.func = None

        self.population = None
        self.funcs = None
        self.child_funcs = None
        self.x = None

        self.sess = None
        self.placeholder = None

    def parse_params(self, params_dict):
        self.f = params_dict['F']
        self.p = params_dict['P']

        if not (params_dict.get('number_of_individuals') is None):
            self.set_number_of_individuals(params_dict['number_of_individuals'])
        else:
            self.set_number_of_individuals(self.dim * 5)

        self.sess = params_dict['sess']
        self.placeholder = params_dict['placeholder']
        self.x = params_dict['x']

    def set_params(self, f, p):
        self.f = f
        self.p = p

    def set_number_of_individuals(self, number_of_individuals):
        self.number_of_individuals = number_of_individuals

    def set_x(self, x):
        self.x = x

    def set_placeholder(self, placeholder):
        self.placeholder = placeholder

    def set_session(self, sess):
        self.sess = sess

    def optimize(self, func, dim, end_cond=None, max_time=None, min_func_value=None):
        start_time = time.time()
        cost_list = []
        self.dim = dim
        self.func = func

        if (self.placeholder is None) and (self.x is None):
            placeholder_dict = {}
        else:
            placeholder_dict = {self.placeholder: self.x}

        population_np = numpy.zeros((self.number_of_individuals, dim))
        for i, _ in enumerate(population_np):
            population_np[i] = numpy.random.uniform(self.min_element, self.max_element, dim)

        self.population = tf.Variable(population_np, tf.double)
        self.sess.run(tf.global_variables_initializer())

        self.funcs = tf.map_fn(func, self.population)
        partners_matrix = tf.cast(tf.random.shuffle(self.population), tf.double)
        a_matrix = tf.cast(tf.random.shuffle(self.population), tf.double)
        b_matrix = tf.cast(tf.random.shuffle(self.population), tf.double)
        mutation_matrix = partners_matrix + self.f * (a_matrix - b_matrix)
        random_matrix = tf.random.uniform((self.number_of_individuals, self.dim))
        mask = tf.cast((random_matrix < self.p), tf.double)
        child_matrix = mask * mutation_matrix - (mask - 1) * self.population
        child_funcs = tf.map_fn(self.func, child_matrix)
        func_mask = tf.cast((child_funcs < self.funcs), tf.double)
        reshaped_func_mask = tf.reshape(func_mask, (self.number_of_individuals, 1))
        pop_ass = self.population.assign(reshaped_func_mask * child_matrix - (reshaped_func_mask - 1) * self.population)

        self.sess.run(tf.global_variables_initializer())
        reason_of_break = 'Maximum iteration reached'
        exit_code = -1
        is_converged = False
        i = 0
        time_list = [0]

        while (end_cond is None) or (i <= end_cond):
            i += 1
            _, funcs = self.sess.run([pop_ass, self.funcs], placeholder_dict)
            cur_cost = numpy.amin(funcs)
            cost_list.append(cur_cost)

            if (not (min_func_value is None)) and (cur_cost <= min_func_value):
                reason_of_break = 'Minimum function value reached'
                exit_code = 0
                is_converged = True
                break

            cur_time = time.time() - start_time
            time_list.append(cur_time)
            if (not (max_time is None)) and (cur_time > max_time):
                reason_of_break = 'Maximum time reached'
                exit_code = -2
                break

        total_time = time.time() - start_time
        time_list.append(total_time)
        funcs = self.sess.run(self.funcs, placeholder_dict)
        min_index = numpy.argmin(funcs)
        misc = {'time_list': time_list}
        result = OptimizeResult(is_converged, i - 1, total_time, cost_list, exit_code,
                                reason_of_break, self.population[min_index], misc=misc)
        return result
