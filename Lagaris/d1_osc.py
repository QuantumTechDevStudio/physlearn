import math
import numpy as np
from numpy.polynomial.hermite import Hermite
from matplotlib import pylab as plt

def wf(n, x):
    shape = []
    if n == 0:
        shape = [1]
    else:
        for i in range(n):
            shape.append(0)
        shape.append(1)
    h_n = Hermite(shape)
    exp_factor = np.exp(-np.square(x)/2)
    #some_factor = (-1)**n/(np.sqrt(2**n *math.factorial(n)*np.sqrt(math.pi)))
    some_factor = 1/(np.sqrt(2**n * math.factorial(n)*np.sqrt(math.pi)))
    res = some_factor*exp_factor*h_n(x)
    return res.reshape(x.size,)

def show_wf(n, x):
    plt.title('Output:')
    plt.grid(True)
    plt.plot(x[0], wf(n,x), "r--")