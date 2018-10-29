import ctypes
import time
import math

import numpy
from tqdm import tqdm

from physlearn.Optimizer.NelderMead import NelderMeadAbstract
from physlearn.Optimizer.OptimizeResult import OptimizeResult


class NelderMeadCtypes(NelderMeadAbstract):
    lib = ctypes.CDLL('/home/andrey/CLionProjects/NelderMead2/build/library.so')

    c_set_simplex = lib.set_simplex
    c_set_params = lib.set_params
    c_iteration = lib.iteration
    c_return_best_point = lib.return_best_point
    c_free_simplex = lib.free_simplex

    def __init__(self, min_element=-1, max_element=1):
        super().__init__(min_element, max_element)
        self.update_iter = -1

    def calc_func(self, params):
        return self.func(params)

    def optimize(self, func, dim, end_cond, min_cost=1e-5):
        # func - оптимизируемая функция, должна принимать numpy.array соотвесвтующей размерности в качесвте параметра
        # dim - размерность функции
        # end_method - условие останова
        # 'variance' - дисперсия набора значений функции симплкса должна быть меньше end_cond
        # 'max_iter' - остановка при достижении end_cond итераций
        self.func = func

        def temp_func(temp_x, temp_dim):
            true_x = numpy.array(temp_x[:temp_dim])
            return self.func(true_x)

        self.dim = dim

        double = ctypes.c_double
        c_func_a = ctypes.CFUNCTYPE(double, ctypes.POINTER(double), ctypes.c_int)
        c_func = c_func_a(temp_func)

        self.x_points = self.create_points()  # Создаем точки
        # Вычисляем значение функции в точках
        self.y_points = numpy.zeros(self.dim + 1)
        for index, x in enumerate(self.x_points):
            self.y_points[index] = self.calc_func(x)

        c_dim = ctypes.c_int(self.dim)
        c_x_points = numpy.ctypeslib.as_ctypes(self.x_points)
        c_y_points = numpy.ctypeslib.as_ctypes(self.y_points)
        c_alpha = double(self.alpha)
        c_beta = double(self.beta)
        c_gamma = double(self.gamma)
        self.c_set_simplex(c_dim, c_x_points, c_y_points)
        self.c_set_params(c_alpha, c_beta, c_gamma)
        self.dot_str = ''
        self.print_str = ''
        self.start_time = time.time()
        prev_update_time = 0

        for i in range(end_cond):
            cur_time = time.time()
            if (cur_time - prev_update_time) >= 1:
                delta = cur_time - self.start_time
                self.speed = i / delta
                self.percent_done = math.floor(i * 100 / end_cond)
                self.update_progress_bar(i)
                prev_update_time = cur_time
            self.c_iteration(c_func)

        best_point = numpy.zeros(self.dim)
        c_best_point = numpy.ctypeslib.as_ctypes(best_point)
        self.c_return_best_point(c_best_point)
        best_point = numpy.ctypeslib.as_array(c_best_point, self.dim)
        end_time = time.time()

        total_time = end_time - self.start_time
        result = OptimizeResult(False, 1, total_time, [0], 0,
                                "HUI", best_point)
        self.c_free_simplex()
        return result
