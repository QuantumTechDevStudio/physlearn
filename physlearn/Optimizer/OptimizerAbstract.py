class OptimizerAbstract:
    params_dict = {}

    def __init__(self, min_element, max_element):
        self.min_element = min_element
        self.max_element = max_element

    def optimize(self, func, dim, end_cond, min_cost=1e-5):
        pass

    def parse_params(self, params_dict):
        pass
