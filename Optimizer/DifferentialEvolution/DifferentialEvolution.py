import time

import numpy

from physlearn.Optimizer.OptimizeResult import OptimizeResult
from physlearn.Optimizer.OptimizerAbstract import OptimizerAbstract


class DifferentialEvolution(OptimizerAbstract):
    amount_of_individuals = None
    f = None
    p = None
    end_method = None
    # Переменные необходимые для работы прогресс бара
    dot_str = ''
    print_str = ''
    start_time = 0
    speed = 0
    update_pb_iter = 1000
    amount_of_dots = 0
    percent_done = 0

    def __init__(self, min_element=-1, max_element=1):
        super().__init__(min_element, max_element)
        self.f = 0.5
        self.p = 0.9

        self.func = None
        self.population = None
        self.func_population = None
        self.child_funcs = []
        self.dim = 0
        self.cost_list = []
        self.end_method = 'max_iter'

        self.update_func = None
        self.update_iter = -1

    def set_number_of_individuals(self, amount_of_individuals):
        self.amount_of_individuals = amount_of_individuals

    def set_params(self, f, p):
        self.f = f
        self.p = p

    def set_end_method(self, end_method):
        self.end_method = end_method

    def parse_params(self, params_dict):
        self.f = params_dict['F']
        self.p = params_dict['P']

        if not (params_dict.get('number_of_individuals') is None):
            self.set_number_of_individuals(params_dict['number_of_individuals'])
        else:
            self.set_number_of_individuals(self.dim * 5)

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
        for index in range(self.amount_of_individuals):
            self.child_funcs[index] = self.func(child_matrix[index])
        # Аналогично, получаем маску для выбора лучшей особей
        func_mask = (self.child_funcs < self.func_population) * 1
        reshaped_func_mask = func_mask.reshape(func_mask.size, 1)
        # Получаем новую популяцию
        self.population = reshaped_func_mask * child_matrix - (reshaped_func_mask - 1) * self.population
        for index in range(self.amount_of_individuals):
            self.func_population[index] = self.func(self.population[index])

    def optimize(self, func, dim, end_cond=None, max_time=None, min_func_value=1e-5):
        # func - оптимизиуемая функция, должна принмать в качетсве параметра массив numpy.array размерности dim
        # dim - размерность
        # amount_of_individuals - количество особей
        # f - сила мутации
        # p - вероятность того, что в потомке элемент будет взят из второго партнера
        self.start_time = time.time()
        self.dim = dim
        self.population = self.create_population()  # Создаем популяцию
        self.func_population = numpy.zeros(self.amount_of_individuals)
        self.func = func
        for index, item in enumerate(self.population):
            self.func_population[index] = self.func(item)
        # Каждый массив: numpy.array([1, 2, ..., amount_of_individuals])

        # каждой особи в популяции значении функции
        self.child_funcs = numpy.zeros(self.amount_of_individuals)
        # Переменные, в которых сохраняется результат
        reason_of_break = 'Maximum iterations reached'  # Причина выхода
        exit_code = -1  # Код выхода
        is_converged = False  # Сошелся алгоритм или нет
        self.cost_list = []  # Список, содержащий значения лучшей функции на каждой итерации

        
        time_list = [0]
        i = 0
        while (end_cond is None) or (i <= end_cond):
            i += 1
            self.iteration()
            cur_cost = numpy.min(self.func_population)
            self.cost_list.append(cur_cost)
            if (not (min_func_value is None)) and (cur_cost <= min_func_value):
                reason_of_break = 'Minimum cost reached'
                exit_code = 0
                is_converged = True
                break

            cur_time = time.time() - self.start_time
            time_list.append(cur_time)
            if (not (max_time is None)) and (cur_time >= max_time):
                reason_of_break = 'Maximum time reached'
                exit_code = -2
                is_converged = False
                break

        end_time = time.time()
        total_time = end_time - self.start_time
        time_list.append(total_time)
        misc = {'time_list': time_list}
        result = OptimizeResult(is_converged, i - 1, total_time, self.cost_list, exit_code,
                                reason_of_break, self.choose_best_individual(), misc=misc)
        return result
