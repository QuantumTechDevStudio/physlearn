import numpy as np
from numpy.linalg import inv
from numpy.polynomial.hermite import hermroots


def norm(matrix):
    norm_matrix = np.sum(np.square(matrix))
    return norm_matrix


def cond(matrix):
    inv_matrix = inv(matrix)
    cond_matrix = norm(matrix)*norm(inv_matrix)
    return cond_matrix


def mean(values):
    res = (1.0/values.size) * np.sum(values)
    return res


def variance(values):
    res = (1.0/values.size) * np.sum(np.square(values - mean(values)))
    return res


def MSE(errors):
    return (1.0/errors.size)*np.sum(np.square(errors))


def std_err(errors):
    return np.sqrt((1.0/(errors.size-1))*np.sum(np.square(errors)))


def monte_carlo(f, args):
    x = args[0][0]
    a = x[0]
    b = x[-1]
    n = x.size
    f_list = f(*args)
    integral = ((b-a)/n)*np.sum(f_list)
    return integral


def root_of_n_hermite(n):
    shape = []
    if n == 0:
        shape = [1]
    else:
        for i in range(n):
            shape.append(0)
        shape.append(1)
    if n!=0:
        return hermroots(shape)
    else:
        return "no roots"


def calc_hermite_grid(amount):
    grid = []
    i = 0
    while amount > len(grid):
        i = i + 1
        set_of_roots = root_of_n_hermite(i)
        for j in range(len(set_of_roots)):
            flag = 0
            for k in range(len(grid)):
                if (abs(grid[k] - set_of_roots[j])) < 5e-2:
                    flag = 1
                    break
            if flag == 0:
                grid.append(set_of_roots[j])
    return np.array(grid)