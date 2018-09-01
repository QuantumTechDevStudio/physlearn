import tensorflow as tf

from physlearn.NeuralNet.NeuralNetAbstract import NeuralNetAbstract


class NeuralNetPro(NeuralNetAbstract):
    placeholders_dict = {}

    def __init__(self, min_element=-1, max_element=1):
        super().__init__(min_element, max_element)

    def calc(self, calc_var, d):
        d.update(self.placeholders_dict)  # Добавляем в словарь d placeholder для матриц весов
        return self.sess.run(calc_var, d)

    def create_tf_matrixes(self):
        tf_matrixes = []
        for index in range(len(self.design) - 1):
            # Цикл идет до длины self.design - 1 потому что, нет ничего после выходного слоя
            current_layer_units = self.design[index][0]
            next_layer_units = self.design[index + 1][0]
            size = current_layer_units * next_layer_units  # Количество элементов матрицы
            weight_breaker = size + self.unroll_breaks[index][1]  # Индекс конца матрицы весов в unroll векторе -
            # ее размер, плюс сдвиг, связанный с предыдущими матрицами
            bias_breaker = weight_breaker + next_layer_units  # Аналогично
            self.unroll_breaks.append((weight_breaker, bias_breaker))
            tf_weight_matrix = tf.placeholder(tf.double)
            tf_bias_vector = tf.placeholder(tf.double)
            tf_matrixes.append((tf_weight_matrix, tf_bias_vector))
            self.size_list.append(((next_layer_units, current_layer_units), (next_layer_units, 1)))
        return tf_matrixes

    def assign_matrixes(self, matrixes):
        for index, layer in enumerate(self.tf_matrixes):
            self.placeholders_dict.update([(layer[0], matrixes[index][0]), (layer[1], matrixes[index][1])])

    def set_random_matrixes(self):
        matrixes = []
        for index in range(len(self.design) - 1):
            # Цикл идет до длины self.design - 1 потому что, нет ничего после выходного слоя
            current_layer_units = self.design[index][0]
            next_layer_units = self.design[index + 1][0]
            weight_matrix, bias_vector = self.create_layer_matrixes((next_layer_units, current_layer_units))
            matrixes.append((weight_matrix, bias_vector))
        self.assign_matrixes(matrixes)
