import numpy
import tensorflow as tf
from physlearn.NeuralNet.NeuralNetAbstract import NeuralNetAbstract
from tqdm import tqdm


class NeuralNet(NeuralNetAbstract):
    train_type = None

    def __init__(self, min_element=-1, max_element=1):
        super().__init__(min_element, max_element)
        self.train_type = None

    def calc(self, calc_var, d):
        return self.sess.run(calc_var, d)

    def create_tf_matrixes(self):
        # Создаем матрицы весов и вектора сдвигов типа tf.Variable
        # Каждый элемент списка tf_matrixes, который представляет из себя (матрица весов, вектор сдвига),
        # отвечает одному слою НС
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
            weight_matrix, bias_vector = self.create_layer_matrixes((next_layer_units, current_layer_units))
            tf_weight_matrix = tf.Variable(weight_matrix, dtype=tf.double)
            tf_bias_vector = tf.Variable(bias_vector, dtype=tf.double)
            tf_matrixes.append((tf_weight_matrix, tf_bias_vector))
        return tf_matrixes

    def train(self, x, y, batch_size, max_iters, alpha):
        # Обучение НС
        # Возвращает список cost_list - список значений ценовой функции на каждой итерации,
        # или -1, если произошла ошибка
        # train_type - тип обучения
        # x - входные данные; x = numpy.array([[...], ...])
        # y - обучающие выходные данные; y = numpy.array([[...], ...])
        # batch_size - размер batch (подвыборки), которая на каждой итерации обучения будет выбираться случайным образом
        # max_iters - количество итераций
        # alpha - шаг обучения
        if self.train_type is None:
            print('Set train type before train')
        if not self.if_compile:  # Проверка на то, что НС была скомпилировнна
            print('Compile model before calculate cost')
            return -1

        self.sess.run(self.init)  # Сброс матриц к начальным значениям
        cost_list = self.__gradient_descent(alpha, batch_size, max_iters, x, y)

        return cost_list

    def continue_train(self, x, y, batch_size, max_iters, alpha):
        # Продолжение обучения
        # Используется тогда, когда цель обучения не была достигнута, но все параметры были выбраны хорошо
        # Олтчается от self.train только тем, что не произоддит сброса и вместо train_type используется self.train_type
        # Параметры - см. self.train
        if not self.if_compile:
            print('Compile model before calculate cost')
            return -1
        cost_list = self.__gradient_descent(alpha, batch_size, max_iters, x, y)
        return cost_list

    def __gradient_descent(self, alpha, batch_size, max_iters, x, y):
        cost_list = []
        optimizer = tf.train.GradientDescentOptimizer(alpha)  # Инициализация оптимизатора...
        train = optimizer.minimize(self.cost)
        # ...которым в данном случае является обычный градиенты спуск с шагом обучения alpha
        for _ in tqdm(range(max_iters)):
            batch_indexes = numpy.random.randint(0, x.shape[1], batch_size)  # Случайный выбор инедксов, из которых
            # будет состоять подвыборка на данной итерации обучения
            _, cur_cost = self.sess.run([train, self.cost], {self.x: x[:, batch_indexes], self.y: y[:, batch_indexes]})
            cost_list.append(cur_cost)
        return cost_list

    def assign_matrixes(self, tf_matrixes):
        # Выполнение присваивания матрицам значений
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!        !!!ВНИМАНИЕ!!!           !!
        # !!    Работает крайне медленно     !!
        # !!Не  рекомендуется к использованию!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        tf_assign = []
        for index, layer in enumerate(self.tf_matrixes):
            tf_assign.append(layer[0].assign(tf_matrixes[index][0]))  # Данная конструкция создает и добавляет в список
            # TensorFlow "присваиватель", который далее запускается через self.sess.run,
            # после чего происходит присваивание
            tf_assign.append(layer[1].assign(tf_matrixes[index][1]))  # Аналогично
            pass
        self.sess.run(tf_assign)  # Присваивание значениям self.tf_matrixes tf_matrixes

    def unroll_matrixes(self):
        # Данная функция "разворачивает" все матрицы в один вектор строку
        numpy_matrixes = self.sess.run(self.tf_matrixes)  # Получение матриц в формате numpy.array
        unroll_vector = self.create_unroll_vector(numpy_matrixes)
        return numpy_matrixes, self.size_list, unroll_vector
