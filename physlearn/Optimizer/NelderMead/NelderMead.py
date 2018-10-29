from physlearn.Optimizer.NelderMead.NelderMeadAbstract import NelderMeadAbstract


class NelderMead(NelderMeadAbstract):

    def calc_func(self, params):
        return self.func(params)
