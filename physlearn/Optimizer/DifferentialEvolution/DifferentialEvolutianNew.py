import numpy
from physlearn.Optimizer.DifferentialEvolution.DifferentialEvolutionAbstract import DifferentialEvolutionAbstract


class DifferentialEvolutionNew(DifferentialEvolutionAbstract):
    alpha = 0
    prev_state = None
    update_func = None
    update_iter = 1
    cur_params = None
    prev_params = None
    # prev_func = None
    grid_changed = False
    # filter_alpha = None
    cost_params = None

    def sigmoid(self, z):
        return 1 / (1 + numpy.exp(-(self.alpha * z)))

    def update(self):
        # print('ok')
        self.grid_changed = True
        self.prev_params = self.cur_params
        self.cur_params = self.update_func()
        self.func_population = numpy.zeros(self.amount_of_individuals)
        cur_func = numpy.zeros_like(self.func_population)
        prev_func = numpy.zeros_like(self.func_population)
        for index in range(self.amount_of_individuals):
            cur_func[index] = self.func(self.population[index], self.cur_params)
            prev_func[index] = self.func(self.population[index], self.prev_params)

        filter_alpha = self.sigmoid((prev_func - cur_func) ** 2)
        self.func_population = filter_alpha * prev_func + (1 - filter_alpha) * cur_func

    def set_alpha(self, q):
        self.alpha = q

    def set_cost_params(self, cost_params):
        self.cost_params = cost_params

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
        cur_child_func = numpy.zeros_like(self.child_func)
        for index in range(self.amount_of_individuals):
            cur_child_func[index] = self.func(child_matrix[index], self.cur_params)
        if self.grid_changed:
            prev_child_func = numpy.zeros_like(self.child_func)
            filter_alpha = numpy.zeros_like(self.child_func)
            for index in range(self.amount_of_individuals):
                prev_child_func[index] = self.func(child_matrix[index], self.prev_params)
                filter_alpha = self.sigmoid((prev_child_func - cur_child_func) ** 2)
            self.child_func = filter_alpha * prev_child_func + (1 - filter_alpha) * cur_child_func
            self.grid_changed = False
        else:
            self.child_func = cur_child_func

        # Аналогично, получаем маску для выбора лучшей особей
        func_mask = (self.child_func < self.func_population) * 1
        reshaped_func_mask = func_mask.reshape(func_mask.size, 1)
        # Получаем новую популяцию
        self.population = reshaped_func_mask * child_matrix - (reshaped_func_mask - 1) * self.population
        # delta = new_de_population - self.population
        # И новый список значений функции особей
        # self.func_population = func_mask * self.child_funcs - (func_mask - 1) * self.func_population
        # self.population = self.alpha * self.prev_state + (1 - self.alpha) * new_de_population

        # for index in range(self.amount_of_individuals):
        #    self.func_population[index] = self.func(self.population[index], self.cur_params)
        self.func_population = func_mask * self.child_func - (func_mask - 1) * self.func_population

        # self.prev_state = delta

    def set_update_func(self, update_func, update_iter=1):
        self.update_func = update_func
        self.update_iter = update_iter

    def optimize(self, func, dim, end_cond, debug_pop_print=-1):
        # func - оптимизиуемая функция, должна принмать в качетсве параметра массив numpy.array размерности dim
        # dim - размерность
        # amount_of_individuals - количество особей
        # f - сила мутации
        # p - вероятность того, что в потомке элемент будет взят из второго партнера
        self.dim = dim
        self.population = self.create_population()  # Создаем популяцию
        self.func = func
        self.cost_list = []

        # Каждый массив: numpy.array([1, 2, ..., amount_of_individuals])
        self.func_population = numpy.zeros(self.amount_of_individuals)
        # self.prev_func = numpy.zeros(self.amount_of_individuals)
        self.prev_state = numpy.zeros_like(self.population)
        self.cur_params = self.update_func()
        # for index in range(self.amount_of_individuals):
        #     self.func_population[index] = self.func(self.population[index])
        # self.func_population = numpy.array(list(map(lambda item: func(item), self.population)))  # Вычисляем для
        # каждой особи в популяции значении функции
        self.child_func = numpy.empty_like(self.func_population)
        if self.end_method == 'max_iter':
            if debug_pop_print == -1:
                for i in tqdm(range(end_cond)):
                    if (i % self.update_iter) == 0:
                        self.update()
                    self.iteration()
                    # self.cost_list.append(numpy.min(self.func_population))
                    _, min_func = self.choose_best_individual()
                    # self.cost_list.append(min_func)
                    self.cost_list.append(numpy.min(self.func_population))
            else:
                for i in tqdm(range(end_cond)):
                    if i % debug_pop_print == 0:
                        print(self.population)
                        print('-------------------------------')
                    if (i % self.update_iter) == 0:
                        self.update()
                    self.iteration()
                    self.cost_list.append(numpy.min(self.func_population))
        return self.choose_best_individual()[0]

    def choose_best_individual(self):
        # Данная функция находит лучшую особь в популяции
        cost_funcs = numpy.zeros_like(self.func_population)
        for index in range(self.amount_of_individuals):
            cost_funcs[index] = self.func(self.population[index], self.cost_params)
        func_list = list(cost_funcs)
        min_func = min(func_list)
        best_index = func_list.index(min_func)
        return self.population[best_index], min_func

    def return_cost_list(self):
        return self.cost_list
