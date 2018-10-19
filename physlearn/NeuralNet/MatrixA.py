class MatrixA:
    matrix = None
    shape = None

    def __init__(self, matrix, shape):
        self.matrix = matrix
        self.shape = shape

    def set_matrix(self, matrix, shape):
        self.matrix = matrix
        self.shape = shape

    def roll_matrix(self, unroll_vector):
        self.matrix = unroll_vector.reshape(self.shape)

    def __mul__(self, x):
        pass
