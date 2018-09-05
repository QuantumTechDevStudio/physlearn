from physlearn.Optimizer.NelderMead.NelderMeadAbstract import NelderMeadAbstract


class NelderMead(NelderMeadAbstract):
    def __init__(self, min_element=-1, max_element=1):
        super().__init__(min_element, max_element)
        self.update_iter = -1

    def calc_func(self, params):
        return self.func(params)
