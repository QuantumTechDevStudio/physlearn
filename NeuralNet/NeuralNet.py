import sys

import tensorflow as tf

from physlearn.NeuralNet.Layer.LayerCreator import LayerCreator


class NeuralNet:
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
        self.min_element = min_element  # Минимальное значение при случайной генерации матриц весов
        self.max_element = max_element  # Максимальное

        self.design = []  # Каждый элемент этого списка хранит в себе либо описание отдельного слоя
        # (количество нейронов, функция активации), либо подсети
        self.amount_of_neurons_in_layer = []  # Список, в котром хранится количество нейронов в каждом слое
        self.activation_funcs = []  # Список с функциями активации
        self.design_len = 0  # Длина self.design
        self.tf_layers = []  # Графы вычислений для каждого слоя
        self.unroll_breaks = [0]  # Границы каждого слоя в развернутом векторе
        self.layers = []  # Здесь хранятся слои, как объекты типа Layer
        self.cur_net_layers = []
        self.assign_list = []
        self.cur_amount_of_sub_nets = 0
        self.number_of_layers = 0

        self.dim = 0  # Размерность развернутого вектора
        self.cur_net_num = 0  # Номер создаваемой независимой НС

        self.if_compile = False  # Была ли НС скомпилированна
        self.layers_created = False
        self.correctness = True  # Все ли корректно в параметрах нейроной сети
        self.numpy_mode = False
        self.assign_list = []
        self.x = None  # Placeholder для входных данных
        self.cost = None  # Переменная ценовой функции
        self.sess = None  # Сессия
        self.init = None  # Начальные значения
        self.output = None  # Переменная выхода НС
        self.outputs = []  # Список, где хранятся тензоры, отвечающие выходам всех независимых НС
        self.train_type = ""  # Тип обучения
        self.amount_of_outputs = None  # Количество выходов
        self.output_activation_func = None  # Функция активации выходов

        self.cost_func = None  # Пользовательская ценовая функция
        self.optimize_params_dict = {}  # Параметры оптимизации

        self.layer_creator = LayerCreator()

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------Методы, которые задают архитектуру НС-------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def add(self, number_of_units, activation_func):
        current_layer = (number_of_units, activation_func)
        self.number_of_layers += 1
        self.cur_amount_of_sub_nets = 0
        self.design.append(current_layer)

    def add_sub_nets(self, number_of_units, activation_funcs):
        if (self.cur_amount_of_sub_nets != 0) and (self.cur_amount_of_sub_nets != len(number_of_units)):
            print('Error in number of sub nets')
        else:
            self.design.append((number_of_units, activation_funcs))
            self.number_of_layers += 1
            self.cur_amount_of_sub_nets = len(number_of_units)

    def add_input_layer(self, amount_of_units):
        self.add(amount_of_units, None)

    def add_output_layer(self, amount_of_units, output_activation_func):
        self.amount_of_outputs = amount_of_units
        self.output_activation_func = output_activation_func
        self.add(self.amount_of_outputs, self.output_activation_func)

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------Создание графа TF и все необходимые для этого методы --------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def compile(self):
        if not self.correctness:
            sys.stderr.write('Error in neural net design')
            return None
        self.sess = tf.Session()  # Создание сессии
        if self.if_compile:  # Проверка, была ли скомпилированна НС ранее...
            # ...если да - сброс к начальным параметрам
            self.layers = []
            self.init = None

        self.if_compile = True
        self.x = tf.placeholder(tf.double)  # Создание placeholder для входных данных...

        for i in range(self.number_of_layers - 1):
            cur_layer_units = self.design[i][0]
            next_layer_units = self.design[i + 1][0]
            activation_func = self.design[i + 1][1]
            cur_layer = self.layer_creator.create_layer(i, cur_layer_units, next_layer_units, activation_func)
            self.unroll_breaks.append(self.unroll_breaks[-1] + cur_layer.return_layer_dim())
            self.layers.append(cur_layer)
        self.dim = self.unroll_breaks[-1]
        self.sess.run(tf.global_variables_initializer())

        self.output = self.__create_output_tensor()

    def __create_output_tensor(self):
        self.tf_layers = [self.layers[0].calc_output(self.x)]
        for i in range(1, self.number_of_layers - 1):
            cur_layer_output = self.layers[i].calc_output(self.tf_layers[-1])
            self.tf_layers.append(cur_layer_output)
        return self.tf_layers[-1]

    def __create_net_from_unroll_vectors(self, unroll_vectors):
        self.layers = []
        for i in range(self.number_of_layers - 1):
            cur_layer = self.layer_creator.create_layer(layer_num=i, layer_unroll_vector=unroll_vectors[i])
            self.layers.append(cur_layer)
        self.output = self.__create_output_tensor()

    def __recreate_net(self):
        self.layers = []
        for i in range(self.number_of_layers - 1):
            cur_layer = self.layer_creator.create_layer(layer_num=i)
            cur_assign = cur_layer.create_assigns()
            self.assign_list.extend(cur_assign)
            self.layers.append(cur_layer)
        self.sess.run(tf.global_variables_initializer())
        self.output = self.__create_output_tensor()

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------Методы, вычисляющие значение НС------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #

    def calc(self, calc_var, x):
        # Метод вычисляет значение любого тензора, так или иначе связанного с выходным тензором НС
        # x - значение placeholder self.x
        return self.sess.run(calc_var, x)

    def run(self, inputs):
        # Метод вычисляет выход НС на входных данных inputs
        result = self.calc(self.output, {self.x: inputs})
        return result

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------Методы, возвращающие различные параметры НС--------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def return_graph(self):
        # Метод возвраащет выходной тензор НС
        return self.output

    def return_session(self):
        # Метод вовращает tf.Session()
        return self.sess

    def return_unroll_dim(self):
        # Метод возвращает размерность unroll_vector
        return self.dim

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------Методы, проводящие манипуляции с матрицами весов------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #

    def enable_numpy_mode(self):
        self.__recreate_net()
        self.numpy_mode = True

    def roll_matrixes(self, unroll_vector):
        if not self.if_compile:
            sys.stderr.write('Compile model before roll matrixes')
            return None

        if self.numpy_mode:
            self.__roll_numpy_matrixes(unroll_vector)
        else:
            self.__roll_tf_matrixes(unroll_vector)

    def __roll_numpy_matrixes(self, unroll_vector):
        placeholder_dict = {}
        for index, _ in enumerate(self.layers):
            left_layer_break = self.unroll_breaks[index]  # Левая граница матрицы весов = правая гранница
            # вектора сдвига предыдущего слоя
            right_layer_break = self.unroll_breaks[index + 1]  # Правая граница матрицы весов =
            # = левая граница вектора сдвига
            layer_unroll_vector = unroll_vector[left_layer_break:right_layer_break]
            placeholder_dict.update(self.layers[index].assign_matrixes(layer_unroll_vector))
        self.sess.run(self.assign_list, placeholder_dict)

    def __roll_tf_matrixes(self, unroll_vector):
        unroll_vectors = []
        for index, _ in enumerate(self.layers):
            left_layer_break = self.unroll_breaks[index]  # Левая граница матрицы весов = правая гранница
            # вектора сдвига предыдущего слоя
            right_layer_break = self.unroll_breaks[index + 1]  # Правая граница матрицы весов =
            # = левая граница вектора сдвига
            layer_unroll_vector = unroll_vector[left_layer_break:right_layer_break]
            unroll_vectors.append(layer_unroll_vector)
        self.__create_net_from_unroll_vectors(unroll_vectors)
