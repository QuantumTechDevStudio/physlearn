import numpy
from physlearn.Optimizer import NelderMeadUtil
from tqdm import tqdm


def optimize(func, dim, end_cond, min_element=-1, max_element=1, end_method='max_iter', alpha=1, beta=0.5, gamma=2):
    # func - оптимизируемая функция, должна принимать numpy.array соотвесвтующей размерности в качесвте параметра
    # dim - размерность функции
    # end_method - условие останова
    # 'variance' - дисперсия набора значений функции симплкса должна быть меньше end_cond
    # 'max_iter' - остановка при достижении end_cond итераций
    x_points = NelderMeadUtil.create_points(dim, min_element, max_element)  # Создаем точки
    y_points = numpy.array(list(map(lambda item: func(item), x_points)))  # Вычисляем значение функции в созданых точках
    if end_method == 'max_iter':  # Если условием выхода является достижение некого числа итераций
        for _ in tqdm(range(end_cond)):
            x_points, y_points = iteration(alpha, beta, func, gamma, x_points, y_points)

    elif end_method == 'variance':  # Условие выхода - дисперсия значений функции не больше заданной величины
        var = end_cond + 1
        while var >= end_cond:
            x_points, y_points = iteration(alpha, beta, func, gamma, x_points, y_points)
            var = NelderMeadUtil.variance(y_points)

    else:
        print('Error in end_method param')
        return -1

    _, _, l_index = NelderMeadUtil.find_points(y_points)  # Определяем точку с нименьшим значением функции
    return x_points[l_index]


def iteration(alpha, beta, func, gamma, x, y):
    x_points = x
    y_points = y
    h_index, g_index, l_index = NelderMeadUtil.find_points(y_points)  # Находим точки h, g и l
    x_center = NelderMeadUtil.calculate_center(x_points, h_index)  # Вычисляем центр масс
    x_reflected = NelderMeadUtil.calculate_reflected_point(x_points[h_index], x_center, alpha)  # Вычисляем отраженную
    # точку
    y_reflected = func(x_reflected)
    # Далее мы делаем ряд действий, в зависимости от соотношения между значениями функции в найденных точках
    # Объяснять подробно нет смысла, так что смотри просто "Метод Нелдера - Мида" в вики
    if y_reflected < y_points[l_index]:
        x_stretch = NelderMeadUtil.calculate_stretched_point(x_reflected, x_center, gamma)
        y_stretch = func(x_stretch)
        if y_stretch < y_points[l_index]:
            x_points[h_index] = x_stretch
            y_points[h_index] = y_stretch
        else:
            x_points[h_index] = x_reflected
            y_points[h_index] = y_reflected

    elif y_reflected <= y_points[g_index]:
        x_points[h_index] = x_reflected
        y_points[h_index] = y_reflected

    else:
        if y_reflected < y_points[h_index]:
            x_points[h_index] = x_reflected
            y_points[h_index] = y_reflected

        x_compress = NelderMeadUtil.calculate_compressed_point(x_points[h_index], x_center, beta)
        y_compress = func(x_compress)
        if y_compress < y_points[h_index]:
            x_points[h_index] = x_compress
            y_points[h_index] = y_compress
        else:
            x_points = NelderMeadUtil.compress_simplex(x_points, x_points[l_index])
            y_points = numpy.array(list(map(lambda item: func(item), x_points)))
    return x_points, y_points
