import numpy

from physlearn.Optimizer.NelderMead.NelderMead import NelderMead


class NelderMeadWQG(NelderMead):
    sigma = 0
    grad_vector = None

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

    def calculate_reflected_point(self):
        self.grad_vector = self.calculate_grad()
        x_reflected = self.x_points[self.l_index] - self.sigma * self.grad_vector
        return x_reflected
