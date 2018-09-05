from physlearn.Optimizer.DifferentialEvolution.DifferentialEvolutionAbstract import DifferentialEvolutionAbstract


class DifferentialEvolution(DifferentialEvolutionAbstract):

    def calc_func(self, params):
        return self.func(params)
