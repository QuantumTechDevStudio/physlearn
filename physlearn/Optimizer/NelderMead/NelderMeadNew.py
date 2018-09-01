from physlearn.Optimizer.NelderMead.NelderMeadAbstract import NelderMeadAbstract


class NelderMeadNew(NelderMeadAbstract):
    cur_params = []

    def calc_func(self, params):
        return self.func(params, self.cur_params)

    def update(self):
        self.cur_params = self.update_func()
        for index, x in enumerate(self.x_points):
            self.y_points[index] = self.calc_func(x)
