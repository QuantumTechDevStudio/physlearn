from physlearn.Optimizer.NelderMead.NelderMeadAbstract import NelderMeadAbstract


class NelderMead(NelderMeadAbstract):
    def __init__(self, min_element=-1, max_element=1):
        super().__init__(min_element, max_element)
        self.update_iter = -1

    def calc_func(self, params):
        return self.func(params)
    # x_points = None
    # y_points = None
    # min_element = -1
    # max_element = 1
    # end_method = 'max_iter'
    # end_cond = None
    # dim = None
    # alpha = 1
    # beta = 0.5
    # gamma = 2
    # func = None
    # h_index, g_index, l_index = None, None, None
    # x_center = None
    # x_reflected = None
    #
    # def __init__(self, min_element=-1, max_element=1):
    #     self.min_element = min_element
    #     self.max_element = max_element
    #     self.end_method = 'max_iter'
    #     self.alpha = 1
    #     self.beta = 0.5
    #     self.gamma = 2
    #     self.method_types = [0, 0, 0, 0]
    #     self.types_list = []
    #     self.cost_list = []
    #
    # def set_params(self, alpha, beta, gamma):
    #     self.alpha = alpha
    #     self.beta = beta
    #     self.gamma = gamma
    #
    # def set_end_method(self, end_method):
    #     self.end_method = end_method
    #
    # def optimize(self, func, dim, end_cond):
    #     # func - оптимизируемая функция, должна принимать numpy.array соотвесвтующей размерности в качесвте параметра
    #     # dim - размерность функции
    #     # end_method - условие останова
    #     # 'variance' - дисперсия набора значений функции симплкса должна быть меньше end_cond
    #     # 'max_iter' - остановка при достижении end_cond итераций
    #     self.func = func
    #     self.dim = dim
    #     self.x_points = self.create_points()  # Создаем точки
    #     self.y_points = numpy.array(list(map(func, self.x_points)))  # Вычисляем значение функции
    #     self.method_types = [0, 0, 0, 0]
    #     self.types_list = []
    #     self.cost_list = []
    #     # в созданых точках
    #     if self.end_method == 'max_iter':  # Если условием выхода является достижение некого числа итераций
    #         for _ in tqdm(range(end_cond)):
    #             method_type = self.iteration()
    #             self.types_list.append(method_type)
    #             self.cost_list.append(numpy.min(self.y_points))
    #             self.method_types[method_type] += 1
    #
    #     elif self.end_method == 'variance':  # Условие выхода - дисперсия значений функции не больше заданной величины
    #         var = end_cond + 1
    #         while var >= end_cond:
    #             method_type = self.iteration()
    #             self.method_types[method_type] += 1
    #             var = NelderMead.variance(self.y_points)
    #
    #     else:
    #         print('Error in end_method param')
    #         return -1
    #
    #     _, _, l_index = self.find_points()  # Определяем точку с нименьшим значением функции
    #     return self.x_points[l_index]
    #
    # def iteration(self):
    #     self.h_index, self.g_index, self.l_index = self.find_points()  # Находим точки h, g и l
    #     self.x_center = self.calculate_center()  # Вычисляем центр масс
    #     self.x_reflected = self.calculate_reflected_point()  # Вычисляем отраженную
    #     # точку
    #     y_reflected = self.func(self.x_reflected)
    #     # Далее мы делаем ряд действий, в зависимости от соотношения между значениями функции в найденных точках
    #     # Объяснять подробно нет смысла, так что смотри просто "Метод Нелдера - Мида" в вики
    #     if y_reflected < self.y_points[self.l_index]:
    #         method_type = 0
    #         x_stretch = self.calculate_stretched_point()
    #         y_stretch = self.func(x_stretch)
    #         if y_stretch < self.y_points[self.l_index]:
    #             self.x_points[self.h_index] = x_stretch
    #             self.y_points[self.h_index] = y_stretch
    #         else:
    #             self.x_points[self.h_index] = self.x_reflected
    #             self.y_points[self.h_index] = y_reflected
    #
    #     elif y_reflected <= self.y_points[self.g_index]:
    #         method_type = 1
    #         self.x_points[self.h_index] = self.x_reflected
    #         self.y_points[self.h_index] = y_reflected
    #
    #     else:
    #         if y_reflected < self.y_points[self.h_index]:
    #             self.x_points[self.h_index] = self.x_reflected
    #             self.y_points[self.h_index] = y_reflected
    #
    #         x_compress = self.calculate_compressed_point()
    #         y_compress = self.func(x_compress)
    #         if y_compress < self.y_points[self.h_index]:
    #             method_type = 2
    #             self.x_points[self.h_index] = x_compress
    #             self.y_points[self.h_index] = y_compress
    #         else:
    #             method_type = 3
    #             self.compress_simplex()
    #             self.y_points = numpy.array(list(map(self.func, self.x_points)))
    #     return method_type
    #
    # def create_points(self):
    #     points = []
    #     for i in range(self.dim + 1):  # Создаем массив точек размера dim + 1 (так требуется по методу)
    #         points.append(numpy.random.uniform(self.min_element, self.max_element, self.dim))
    #     return numpy.array(points)
    #
    # def find_points(self):
    #     # В данном методе мы находим три точки: h - точка с наибользим значением оптимизируемой функции,
    #     # g - следующая за точкой с наибольишм значение, l - с наименьшим значением.
    #     # Далее мы задаем начальные парамтры...
    #     h_point = -sys.maxsize
    #     g_point = -sys.maxsize
    #     l_point = sys.maxsize
    #     h_index = 0
    #     g_index = 0
    #     l_index = 0
    #     # ...и проводим стандарнтый поиск.
    #     for index, item in enumerate(self.y_points):
    #         if item > h_point:
    #             g_point = h_point
    #             h_point = item
    #             g_index = h_index
    #             h_index = index
    #         elif (item > g_point) and (item != h_point):
    #             g_point = item
    #             g_index = index
    #         if item < l_point:
    #             l_point = item
    #             l_index = index
    #
    #     return h_index, g_index, l_index
    #
    # def calculate_center(self):
    #     # Вычисляем "центр масс" всех точек, за исключением h
    #     sum_data = 0
    #     n = len(self.x_points) - 1
    #     for index, item in enumerate(self.x_points):
    #         if index != self.h_index:
    #             sum_data += item
    #
    #     return sum_data / n
    #
    # def calculate_reflected_point(self):
    #     # В данной функции выполняется отражение точки h относительно центра масс
    #     x_h = self.x_points[self.h_index]
    #     x_reflected = ((1 + self.alpha) * self.x_center) - (self.alpha * x_h)
    #     return x_reflected
    #
    # def calculate_stretched_point(self):
    #     # В данной функции выполняется растяжение в направлении, соединяющим h, center и reflected
    #     x_stretch = (self.gamma * self.x_reflected) + ((1 - self.gamma) * self.x_center)
    #     return x_stretch
    #
    # def calculate_compressed_point(self):
    #     x_h = self.x_points[self.h_index]
    #     # В данной функции выполняется сжатие к center
    #     x_compress = (self.beta * x_h) + ((1 - self.beta) * self.x_center)
    #     return x_compress
    #
    # def compress_simplex(self):
    #     # В данной функции происходит схатие всего симплекса к l
    #     x_l = self.x_points[self.l_index]
    #     for index, x_i in enumerate(self.x_points):
    #         self.x_points[index] = 0.5 * (x_i + x_l)
    #
    # def return_method_types(self):
    #     return self.method_types
    #
    # def return_types_list(self):
    #     return self.types_list
    #
    # def return_cost_list(self):
    #     return self.cost_list
    #
    # @staticmethod
    # def average(data):
    #     # Вычисление среднего значения
    #     return sum(data) / len(data)
    #
    # @staticmethod
    # def variance(data):
    #     # Вычисление дисперсии
    #     mean_data = NelderMead.average(data)
    #     sum_var = 0
    #     for item in data:
    #         sum_var += (item - mean_data) ** 2
    #     return sum_var / len(data)
