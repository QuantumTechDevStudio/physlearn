from physlearn.Optimizer.NelderMead.NelderMeadAbstract import NelderMeadAbstract


class NelderMead(NelderMeadAbstract):

    def calculate_reflected_point(self):
        # В данной функции выполняется отражение точки h относительно центра масс
        x_h = self.x_points[self.h_index]
        x_reflected = ((1 + self.alpha) * self.x_center) - (self.alpha * x_h)
        return x_reflected

