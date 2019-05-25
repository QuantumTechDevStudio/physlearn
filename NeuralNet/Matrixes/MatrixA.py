class MatrixA:

    def __init__(self, matrix, shape, matrix_type=0):
        self.matrix_assign = None
        self.matrix_placeholder = None
        self.shape = shape
        if matrix_type == 0:
            self.matrix = matrix
        elif matrix_type == 1:
            self.roll_matrix(matrix)

    def set_matrix(self, matrix, shape):
        self.matrix = matrix
        self.shape = shape

    def roll_matrix(self, unroll_vector):
        pass

    def create_assigns(self):
        pass

    def assign_matrixes(self, unroll_vector):
        pass

    def __mul__(self, x):
        pass
