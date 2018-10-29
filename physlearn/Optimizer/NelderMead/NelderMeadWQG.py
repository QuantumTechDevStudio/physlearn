import numpy
from physlearn.Optimizer.NelderMead.NelderMead import NelderMeadAbstract
from physlearn.Optimizer.Pauell import Pauell
from physlearn.Optimizer.NelderMead import NelderMead


class NelderMeadWQG(NelderMeadAbstract):
    sigma = 0
    grad_vector = None
    p = Pauell(1, 0.1, 0.1)
    nm = NelderMead(progress_bar=False)

    def calc_func(self, params):
        return self.func(params)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def calculate_grad(self):
        xs = self.x_points.diagonal()
        f_xs = self.func(xs)
        grad_vector = numpy.zeros(self.dim)
        for index in range(self.x_points.shape[1]):
            if ((index + 1) % 2) == 0:
                grad_vector[index] = (self.y_points[index - 1] - f_xs) / (self.x_points[index - 1][index] - xs[index])
            else:
                grad_vector[index] = (self.y_points[index + 1] - f_xs) / (self.x_points[index + 1][index] - xs[index])
        # print(grad_vector)
        return grad_vector

    def find_best_sigma(self):
        def temp_func(sigma):
            return self.calc_func(self.x_points[self.l_index] - sigma * self.grad_vector)
        res = self.nm.optimize(temp_func, 1, 1000)
        return res.x

    def calculate_reflected_point(self):
        self.grad_vector = self.calculate_grad()
        sigma = self.find_best_sigma()
        x_reflected = self.x_points[self.l_index] - sigma * self.grad_vector
        return x_reflected
