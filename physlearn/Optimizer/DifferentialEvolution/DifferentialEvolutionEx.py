import numpy

from physlearn.Optimizer.DifferentialEvolution.DifferentialEvolutionAbstract import DifferentialEvolutionAbstract


class DifferentialEvolutionEx(DifferentialEvolutionAbstract):
    update_func = None
    update_iter = 1
    cur_params = None

    def update(self):
        self.cur_params = self.update_func()
        self.func_population = numpy.zeros(self.amount_of_individuals)
        for index in range(self.amount_of_individuals):
            self.func_population[index] = self.calc_func(self.population[index])

    def calc_func(self, params):
        return self.func(params, self.cur_params)
