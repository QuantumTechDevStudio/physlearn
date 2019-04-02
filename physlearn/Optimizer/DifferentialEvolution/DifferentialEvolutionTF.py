import numpy
import tensorflow as tf
from tqdm import tqdm

tfe = tf.contrib.eager


class DifferentialEvolutionTF:
    def __init__(self):
        self.f = 0.5
        self.p = 0.9
        self.dim = -1
        self.number_of_individuals = -1
        self.func = None

        self.population = None
        self.funcs = None
        self.child_funcs = None
        self.x = None

    def set_params(self, f, p):
        self.f = f
        self.p = p

    def set_number_of_individuals(self, number_of_individuals):
        self.number_of_individuals = number_of_individuals

    def set_x(self, x):
        self.x = x

    def optimize(self, func, dim, end_cond, placeholder=None, session=None):
        cost_list = []
        sess = session
        self.dim = dim
        self.func = func
        population_np = numpy.zeros((self.number_of_individuals, dim))
        for i, _ in enumerate(population_np):
            population_np[i] = numpy.random.uniform(-1, 1, dim)

        self.population = tf.Variable(population_np, tf.double)

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

        sess.run(tf.global_variables_initializer())
        for _ in tqdm(range(end_cond)):
            #print(self.funcs)
            _, funcs = sess.run([pop_ass, self.funcs], {placeholder: self.x})
            funcs = funcs
            cur_cost = min(funcs)
            cost_list.append(cur_cost)

        funcs = list(sess.run(self.funcs, {placeholder: self.x}))
        min_cost = min(funcs)
        min_index = funcs.index(min_cost)
        return self.population[min_index], cost_list
