import tensorflow as tf

from physlearn.Optimizer.DifferentialEvolution import DifferentialEvolution
from physlearn.Optimizer.DifferentialEvolution import DifferentialEvolutionTF
from physlearn.Optimizer.NelderMead import NelderMead

class Optimizer:
    nelder_mead = 'nelder-mead'
    diff_evolution = 'diff evolution'
    diff_evolution_tf = 'diff evolution tf'
    
    def __init__(self, min_element=-1, max_element=1):
        self.params_dict = {'alpha': 1,
                            'beta': 0.5,
                            'gamma': 2,
                            'epsilon': 0.5,
                            'search_depth': 100,
                            'F': 0.5,
                            'P': 0.9,
                            'number_of_individuals': None,
                            'sess': tf.Session(),
                            'placeholder': None,
                            'x': None}

        self.min_element = min_element
        self.max_element = max_element

        self.misc_funcs = []
        self.misc_names = []



    def add_misc_func(self, func, name):
        self.misc_funcs.append(func)
        self.misc_names.append(name)

    def optimize(self, func, dim, optimizer='Nelder-Mead', end_cond=None, max_time=None, min_func_value=None):
        if self.params_dict['number_of_individuals'] is None:
            self.params_dict['number_of_individuals'] = int(dim * 7.5)

        if optimizer.lower() == 'nelder-mead':
            opt = NelderMead(self.min_element, self.max_element)
            for index, misc_func in enumerate(self.misc_funcs):
                name = self.misc_names[index]
                opt.add_misc_func(misc_func, name)

        elif optimizer.lower() == 'diff evolution':
            opt = DifferentialEvolution(self.min_element, self.max_element)

        elif optimizer.lower() == 'diff evolution tf':
            opt = DifferentialEvolutionTF(self.min_element, self.max_element)

        else:
            print('Unknown optimizer')
            return -1

        opt.parse_params(self.params_dict)
        res = opt.optimize(func, dim, end_cond, max_time, min_func_value)
        return res
