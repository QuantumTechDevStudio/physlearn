import math


def rastrigin(x, n):
    A = 10
    sum_part = 0
    for item in x:
        sum_part += item ** 2 - A * math.cos(2 * math.pi * item)
    return A * n + sum_part


def rosenbrock(x, n):
    s = 0
    l = len(x)
    for i in range(l - 1):
        s += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return s


def himmelblau(x):
    x, y = x[0], x[1]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def cross_in_tray(x):
    x, y = x[0], x[0]
    return -0.0001 * abs(math.sin(x) * math.sin(y) * math.exp(abs(100 -
                                                                  math.sqrt(x ** 2 + y ** 2) / math.pi)) + 1) ** 0.1
