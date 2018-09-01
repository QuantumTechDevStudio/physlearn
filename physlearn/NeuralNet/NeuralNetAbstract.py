import xml.etree.ElementTree as Tree

import numpy
import tensorflow as tf


class NeuralNetAbstract:
    design = []  # Слои нейронной сети в формате (количество нейронов, функция активации)
    if_compile = False  # Была ли НС скомпилированна
    x = None  # Placeholder для входных данных
    y = None  # Placeholder для обучающих выходных данных
    cost = None  # Переменная ценовой функции
    sess = None  # Сессия
    init = None  # Начальные значения
    tf_matrixes = None  # Матрицы слоев типа tf.Variable в формате (веса слоя, сдвиг) (см. self.__create_tf_matrixes)
    tf_layers = None  # Значения после каждого слоя (см self.compile)
    output = None  # Переменная выхода НС
    train_type = ""  # Тип обучения
    size_list = []  # Размеры матриц весов и векторов сдвига
    unroll_breaks = [(0, 0)]  # Индексы концов каждой матрицы в unroll векторе
    amount_of_outputs = None  # Количество выходов
    output_activation_func = None  # Функция активации выходов
    # Минмальный и максимальные элементы, которые могут быть сгенерированны при случайной инициализации матриц весов
    min_element = -1
    max_element = 1

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------Здесь задаются стандартные функции активации---------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Просто тождественная функция
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def sigmoid(x):
        return tf.sigmoid(x)

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------------Конструктор---------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def __init__(self, min_element=-1, max_element=1):
        # Присваивание значений, сброс к начальным условиям
        self.min_element = min_element
        self.max_element = max_element
        self.design = []
        self.tf_matrixes = []
        self.tf_layers = []
        self.size_list = []
        self.unroll_breaks = [(0, 0)]

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------Методы, которые задают архитектуру НС-------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def add(self, amount_of_units, activation_func):
        # Добавление еще одного слоя в НС
        # Слой добавляется в self.design в необходимом формате
        current_layer = (amount_of_units, activation_func)
        self.design.append(current_layer)

    def add_layer_ex(self):
        pass

    def add_input_layer(self, amount_of_units):
        # Добавление входного слоя
        # Функция активации входного слоя нигде не используется, но указывается self.linear,
        # как наследие прошлой реализации, хотя можно использовать и None
        self.add(amount_of_units, self.linear)

    def add_output_layer(self, amount_of_units, output_activation_func):
        self.amount_of_outputs = amount_of_units
        self.output_activation_func = output_activation_func

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------Загрузка НС из файла-------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def load_net_from_file(self, filename):
        func_dict = {'sigmoid': self.sigmoid, 'linear': self.linear}
        net_xml = Tree.parse(filename)
        net_root = net_xml.getroot()
        for layer in net_root:
            layer_type = layer.tag
            if layer_type == 'input_layer':
                amount_of_neurons = int(layer.attrib['amount_of_neurons'])
                self.add_input_layer(amount_of_neurons)
            elif layer_type == 'output_layer':
                amount_of_neurons = int(layer.attrib['amount_of_neurons'])
                activation_func = layer.attrib['activation']
                self.add_output_layer(amount_of_neurons, func_dict[activation_func])
            else:
                amount_of_neurons = int(layer.attrib['amount_of_neurons'])
                activation_func = layer.attrib['activation']
                self.add(amount_of_neurons, func_dict[activation_func])

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------Создание графа TF и все необходимые для этого методы --------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def compile(self):
        # Здесь создается граф вычислений НС
        self.sess = tf.Session()  # Создание сессии
        if self.if_compile:  # Проверка, была ли скомпилированна НС ранее...
            # ...если да - сброс к начальным параметрам
            self.tf_matrixes = []
            self.tf_layers = []
            self.init = None
        else:
            self.add(self.amount_of_outputs, self.output_activation_func)  # Добавление выходного слоя
            # Выходной слой добавляется здесь, так как необходиом гарантировать, что он является последним
        self.if_compile = True
        self.x = tf.placeholder(tf.double)  # Создание placeholder для входных данных...
        self.y = tf.placeholder(tf.double)  # ...и обучающих выходов

        self.tf_matrixes = self.create_tf_matrixes()  # Создание матриц
        self.init = tf.global_variables_initializer()  # Инициализатор переменных
        self.sess.run(self.init)  # Инициализация переменных
        for index, layer in enumerate(self.tf_matrixes):
            weight = layer[0]  # Матрица весов
            bias = layer[1]  # Вектор сдвига
            activation_func = self.design[index + 1][1]  # Функция активации берется из (!!!) СЛЕДУЮЩЕГО (!!!) слоя
            # Этот определенный "костыль", связан с тем, что функция активации является "входным" параметром для слоя =>
            # функция активации, действующая на данные "между" входным и первым слоем хранится в первом слое
            if index == 0:
                current_layer = activation_func(tf.matmul(weight, self.x) + bias)  # Если это первый слой, то действуем
                # на входные данные...
            else:
                prev_layer = self.tf_layers[index - 1]  # ...если нет - то получаем выход с предыдущего слоя...
                current_layer = activation_func(tf.matmul(weight, prev_layer) + bias)  # ...и действуем на него
            self.tf_layers.append(current_layer)
        self.output = self.tf_layers[-1]  # Выход нейронной сети - это последний слой => послдений элемент tf_layers

    def create_tf_matrixes(self):
        # Данный метод создает матрицы в формате tf.placeholder
        return []

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------Методы, вычисляющие значение НС------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #

    def calc(self, calc_var, d):
        # Данный метод проводит вычисление TF величины calc_var с параметрами d
        # calc_var - TF Tensor
        # d - словарь, такого типа, который требуется в sess.run
        return []

    def run(self, inputs):
        # Вычисление результата работы НС на выходных данных inputs
        # inputs = numpy.array([[...], ...])
        result = self.calc(self.output, {self.x: inputs})
        return result

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------Методы, упрощающие работу со стнадартными ценовыми функциями-------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def set_train_type(self, train_type):
        # В данной функции устанавливается тип обучения и выражение для вычисления ценовой функции
        self.train_type = train_type
        self.__set_cost_function()

    def __set_cost_function(self):
        # В зависимости от типа обучения, выбираем нужную ценовую функцию
        if self.train_type == 'prediction':
            self.cost = tf.reduce_mean((self.output - self.y) ** 2)  # Вычисление ценовой функции
        elif self.train_type == 'logistic':
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output) +
                                                      (1 - self.y) * tf.log(1 - self.output)))

    def calculate_cost(self, x, y):
        # Вычисление ценовой функции
        # Параметры x и y - аналогично self.train
        if not self.if_compile:
            print('Compile model before calculate cost')
            return -1

        if self.cost is None:  # Проверка на то, что self.cost не None...
            self.__set_cost_function()

        return self.calc(self.cost, {self.x: x, self.y: y})  # Возвращаем вычисленное значение ценовой функции

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------Методы, возвращающие различные параметры НС--------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def return_graph(self):
        # Возвращает TF Tensor, отвечающий выходному слою
        return self.output

    def return_session(self):
        # Возвращет TF session
        return self.sess

    def return_unroll_dim(self):
        # Возвращает размерность "развернутого" вектора
        return self.unroll_breaks[-1][-1]

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------Методы, проводящие манипуляции с матрицами весов------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #

    def create_layer_matrixes(self, size):
        # Создаем numpy матрицы слоя
        weight_matrix = numpy.random.uniform(self.min_element, self.max_element, size)
        # Вектор сдвига (bias) должен иметь строк столько же, сколько матрица весов (weight_matrix), и один столбец
        # (см. "Broadcast")
        bias_vector = numpy.random.uniform(self.min_element, self.max_element, (size[0], 1))
        self.size_list.append((weight_matrix.shape, bias_vector.shape))
        return weight_matrix, bias_vector

    def unroll_matrixes(self):
        # Данная функция "разворачивает" все матрицы в один вектор строку
        pass

    def assign_matrixes(self, matrixes):
        return []

    def roll_matrixes(self, unroll_vector):
        # Противоположно self.unroll_matrixes, roll_matrixes сворачивает матрицы из вектора обратно в нормальный вид
        tf_matrixes = []
        for index, layer in enumerate(self.unroll_breaks[1:]):
            left_weight_break = self.unroll_breaks[index][1]  # Левая граница матрицы весов = правая гранница
            # вектора сдвига предыдущего слоя
            right_weight_break = layer[0]  # Правая граница матрицы весов = левая граница вектора сдвига
            right_bias_break = layer[1]  # Правая граница вектора сдвига
            # Далее мы выделяем нужный нам фрагмент из развернутого вектра, и делаем его нужным размером
            weight_matrix = unroll_vector[left_weight_break:right_weight_break].reshape(self.size_list[index][0])
            bias_vector = unroll_vector[right_weight_break:right_bias_break].reshape(self.size_list[index][1])
            tf_matrixes.append((weight_matrix, bias_vector))
        self.assign_matrixes(tf_matrixes)

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------Прочее-------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def init_params(self):
        # Инициализация начальных параметров
        self.sess.run(self.init)

    @staticmethod
    def create_unroll_vector(numpy_matrixes):
        # Данный метод "разворачивает" матрицы в один вектор строку
        unroll_vector = numpy.empty(0)  # Создаем пустой вектор, в который будем добавлять развернутые матрицы
        for layer in numpy_matrixes:
            weight_matrix = layer[0]
            bias_vector = layer[1]
            unroll_vector = numpy.append(unroll_vector, weight_matrix)
            unroll_vector = numpy.append(unroll_vector, bias_vector)
        return unroll_vector
