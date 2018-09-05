import sys
import time
import math

import numpy

from physlearn.Optimizer.OptimizeResult import OptimizeResult


class DifferentialEvolutionAbstract:
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
        self.min_element = min_element
        self.max_element = max_element
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

    def set_amount_of_individuals(self, amount_of_individuals):
        self.amount_of_individuals = amount_of_individuals

    def set_params(self, f, p):
        self.f = f
        self.p = p

    def set_end_method(self, end_method):
        self.end_method = end_method

    def update(self):
        pass

    def set_update_func(self, update_func, update_iter=1):
        self.update_func = update_func
        self.update_iter = update_iter

    def calc_func(self, params):
        return []

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
        # print(self.cur_params)
        # Создаем необходимые матрицы, перемешиванием матрицы популяции
        # cur_params = self.update_func()
        # self.func_population = numpy.zeros(self.amount_of_individuals)
        # for index in range(self.amount_of_individuals):
        #    self.func_population[index] = self.func(self.population[index], cur_params)
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
            self.child_funcs[index] = self.calc_func(child_matrix[index])
        # Аналогично, получаем маску для выбора лучшей особей
        func_mask = (self.child_funcs < self.func_population) * 1
        reshaped_func_mask = func_mask.reshape(func_mask.size, 1)
        # Получаем новую популяцию
        self.population = reshaped_func_mask * child_matrix - (reshaped_func_mask - 1) * self.population
        for index in range(self.amount_of_individuals):
            self.func_population[index] = self.calc_func(self.population[index])

    def optimize(self, func, dim, end_cond, min_cost=1e-5):
        # func - оптимизиуемая функция, должна принмать в качетсве параметра массив numpy.array размерности dim
        # dim - размерность
        # amount_of_individuals - количество особей
        # f - сила мутации
        # p - вероятность того, что в потомке элемент будет взят из второго партнера
        self.dim = dim
        self.population = self.create_population()  # Создаем популяцию
        self.func_population = numpy.zeros(self.amount_of_individuals)
        self.func = func
        for index, item in enumerate(self.population):
            self.func_population[index] = self.calc_func(item)
        self.update()
        # Каждый массив: numpy.array([1, 2, ..., amount_of_individuals])

        # каждой особи в популяции значении функции
        self.child_funcs = numpy.zeros(self.amount_of_individuals)
        # Переменные, в которых сохраняется результат
        reason_of_break = ''  # Причина выхода
        amount_of_iterations = 0  # Количество выполненных итераций
        exit_code = -100  # Код выхода
        is_converged = False  # Сошелся алгоритм или нет
        self.cost_list = []  # Список, содержащий значения лучшей функции на каждой итерации

        self.dot_str = ''
        self.print_str = ''

        if self.update_iter < 0:
            self.update_iter = end_cond + 1

        self.start_time = time.time()
        for i in range(end_cond):
            if i != 0:
                cur_time = time.time()
                delta = cur_time - self.start_time
                self.speed = i / delta
                self.update_pb_iter = math.ceil(self.speed * 1)
                self.percent_done = math.floor(i * 100 / end_cond)

            if (i % self.update_pb_iter) == 0:
                self.update_progress_bar(i)

            if (i % self.update_iter) == 0:
                self.update()
            self.iteration()
            cur_cost = numpy.min(self.func_population)
            self.cost_list.append(cur_cost)
            if cur_cost <= min_cost:
                reason_of_break = 'Minimum cost reached'
                exit_code = 0
                amount_of_iterations = i
                is_converged = True
                break

            if i == (end_cond - 1):
                reason_of_break = 'Maximum iterations reached'
                exit_code = -1
                amount_of_iterations = i + 1
                is_converged = False

        end_time = time.time()
        total_time = end_time - self.start_time
        self.percent_done = 100
        self.update_progress_bar(end_cond)
        result = OptimizeResult(is_converged, amount_of_iterations, total_time, self.cost_list, exit_code,
                                reason_of_break, self.choose_best_individual())
        return result

    def update_progress_bar(self, i):
        eraser = ''.ljust(len(self.print_str))
        sys.stderr.write('\r' + eraser)
        if self.amount_of_dots == 4:
            self.dot_str = ''
            self.amount_of_dots = 0
        self.dot_str += '.'
        self.amount_of_dots += 1
        speed_str = '{:.3f}'.format(self.speed)
        self.print_str = self.dot_str.ljust(5) + str(i) + ' (' + str(self.percent_done) + '%) ' + speed_str + ' it\s'
        sys.stderr.write('\r' + self.print_str)
