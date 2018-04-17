import numpy
from numba import jit, prange
from tqdm import tqdm


class DifferentialEvolution:
    def __init__(self, amount_of_individuals, end_cond, end_method='max_iter', f=0.5, p=0.9, min_element=-1,
                 max_element=1):
        self.amount_of_individuals = amount_of_individuals
        self.end_cond = end_cond
        self.end_method = end_method
        self.f = f
        self.p = p
        self.min_element = min_element
        self.max_element = max_element

        self.func = None
        self.population = None
        self.func_population = None
        self.dim = 0
        self.child_funcs = None
        self.cost_list = []

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
        # Создаем необходимые матрицы, перемешиванием матрицы популяции
        partners_matrix = numpy.random.permutation(self.population)
        a_matrix = numpy.random.permutation(self.population)
        b_matrix = numpy.random.permutation(self.population)
        # Мутировавший партнер вычисляется по соотвествующей формуле
        mutation_matrix = partners_matrix + self.f * (a_matrix - b_matrix)
        # Далее мы создаем "маску". Если на месте с инедксами i, j  в маске стоит единица, то соотвествующий
        # элемент потомка
        # берется из мутировавшего партнера. Если 0 - то из исходного.
        # Для начала создаем случайную матрицу, заполненную числами от 0 до 1 с равномерным распределением
        random_matrix = numpy.random.random(self.population.shape)
        # Затем сравниваем эту матрицу с нужной вероятноостью выпадения единицы. После сравнения у нас получится матрица
        # каждый элемент которой есть булевская переменная, причем значения True будут в ней находится с вероятностью p,
        # а False - 1-p. Затем, после домножения на 1 True превратится в единиуц, а False в ноль.
        mask = (random_matrix < self.p) * 1
        # Затем мы получаем матрицу потомков
        child_matrix = mask * mutation_matrix - (mask - 1) * self.population
        # Вычисляем значения оптимизируемой функции на потомках
        # child_funcs = numpy.array(list(map(self.func, child_matrix)))
        for index in prange(self.amount_of_individuals):
            self.child_funcs[index] = self.func(child_matrix[index])

        # Аналогично, получаем маску для выбора лучшей особей
        func_mask = (self.child_funcs < self.func_population) * 1
        reshaped_func_mask = func_mask.reshape(func_mask.size, 1)
        # Получаем новую популяцию
        self.population = reshaped_func_mask * child_matrix - (reshaped_func_mask - 1) * self.population
        # И новый список значений функции особей
        self.func_population = func_mask * self.child_funcs - (func_mask - 1) * self.func_population

    def optimize(self, func, dim, debug_pop_print=-1):
        # func - оптимизиуемая функция, должна принмать в качетсве параметра массив numpy.array размерности dim
        # dim - размерность
        # amount_of_individuals - количество особей
        # f - сила мутации
        # p - вероятность того, что в потомке элемент будет взят из второго партнера
        self.dim = dim
        self.population = self.create_population()  # Создаем популяцию
        self.func = func
        # Каждый массив: numpy.array([1, 2, ..., amount_of_individuals])
        self.func_population = numpy.zeros(self.amount_of_individuals)
        for index in prange(self.amount_of_individuals):
            self.func_population[index] = self.func(self.population[index])
        # self.func_population = numpy.array(list(map(lambda item: func(item), self.population)))  # Вычисляем для
        # каждой особи в популяции значении функции
        self.child_funcs = numpy.empty_like(self.func_population)
        if self.end_method == 'max_iter':
            if debug_pop_print == -1:
                for _ in tqdm(range(self.end_cond)):
                    self.iteration()
                    self.cost_list.append(numpy.min(self.func_population))
            else:
                for i in tqdm(range(self.end_cond)):
                    if i % debug_pop_print == 0:
                        print(self.population)
                        print('-------------------------------')
                    self.iteration()
                    self.cost_list.append(numpy.min(self.func_population))
        return self.choose_best_individual()

    def return_cost_list(self):
        return self.cost_list
