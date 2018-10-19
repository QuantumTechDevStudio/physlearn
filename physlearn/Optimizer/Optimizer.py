from physlearn.Optimizer import OptimizeResult
from physlearn.Optimizer.DifferentialEvolution import DifferentialEvolution
from physlearn.Optimizer.NelderMead import NelderMead


def optimize(params_dict, func, dim, end_cond, min_cost=1e-5):
    method = params_dict['method']
    res = OptimizeResult(False, 0, 0, [-100], -10, 'Unknown method', [])
    if method == 'nelder-mead':
        nm = NelderMead()
        nm.parse_params(params_dict)
        res = nm.optimize(func, dim, end_cond, min_cost=min_cost)

    if method == 'diff evolution':
        de = DifferentialEvolution()
        de.parse_params(params_dict)
        res = de.optimize(func, dim, end_cond, min_cost=min_cost)

    return res
