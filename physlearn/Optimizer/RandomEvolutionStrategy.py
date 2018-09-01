import numpy
from tqdm import tqdm


def create_vectors(dim, sigma, amount_of_vectors):
    vectors = numpy.array([numpy.zeros(dim) for _ in range(amount_of_vectors)])
    for i in range(amount_of_vectors):
        vectors[i] = numpy.random.normal(0, sigma, dim)
    return vectors


def optimize(func, dim, sigma, amount_of_vectors, max_iters, alpha, x0=None, min_element=-1, max_element=1):
    if x0 is None:
        x = numpy.random.uniform(min_element, max_element, dim)
    else:
        x = x0
    cost_list = []
    for _ in tqdm(range(max_iters)):
        random_vectors = create_vectors(dim, sigma, amount_of_vectors)
        sum_vector = numpy.zeros_like(x)
        for vector in random_vectors:
            sum_vector += vector * func(x + vector)
        #print(sum_vector / amount_of_vectors)
        gradient = (1 / (sigma ** 2)) * sum_vector / amount_of_vectors
        #print(gradient)
        x -= alpha * gradient
        cost_list.append(func(x))
    return cost_list, x
