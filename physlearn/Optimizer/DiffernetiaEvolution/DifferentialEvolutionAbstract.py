import numpy


class DifferentialEvolutionAbstract:
    amount_of_individuals = None
    f = None
    p = None
    end_method = None

    def __init__(self, min_element=-1, max_element=1):
        self.min_element = min_element
        self.max_element = max_element
        self.f = 0.5
        self.p = 0.9

        self.func = None
        self.population = None
        self.func_population = None
        self.dim = 0
        self.child_func = None
        self.cost_list = []
        self.end_method = 'max_iter'

    def set_amount_of_individuals(self, amount_of_individuals):
        self.amount_of_individuals = amount_of_individuals

    def set_params(self, f, p):
        self.f = f
        self.p = p

    def set_end_method(self, end_method):
        self.end_method = end_method

    def create_population(self):
        # Создаем популяцию
        population = []
        for _ in range(self.amount_of_individuals):
            population.append(numpy.random.uniform(self.min_element, self.max_element, self.dim))
        return numpy.array(population)

    def choose_best_individual(self):
        # Данная функция находит лучшую особь в популяции
        func_list = list(self.func_population)
        best_index = func_list.index(min(func_list))
        return self.population[best_index]

    def iteration(self):
        return []

    def optimize(self, func, dim, end_cond, debug_pop_print=-1):
        return []

    def return_cost_list(self):
        return self.cost_list
