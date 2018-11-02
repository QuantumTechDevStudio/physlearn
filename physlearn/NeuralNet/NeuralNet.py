import sys
import xml.etree.ElementTree as Tree

import numpy
import tensorflow as tf

from physlearn.NeuralNet.Layer.LayerBlocks import LayerBlocks
from physlearn.NeuralNet.Layer.LayerFC import LayerFC
from physlearn.Optimizer.Optimizer import optimize


class NeuralNet:
    """Neural net class. Implements the main function to create, compile, train and run feed-forward neural networks.

    Main methods:

    add_input_layer - add input layer to NN

    add - add layer to neural network

    add_output_layer - add output layer to neural network

    add_sub_nets - add sub nets to neural network

    is_correct - verifies the teh sub nets parameters are correct

    load_net_from_file - load neural net from special XML file

    compile - compile neural networks. All next methods doesn't work without compiled NN

    run - calc output of neural network on some inputs data

    calc - calc any tf.Tensor which depends of neural networks

    return_graph - returns tf.Tensor which respect to output layer

    return_session - returns tf.Session

    return_unroll_dim - returns dim of unroll vector

    roll_matrixes - roll vector into weight matrixes

    """

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
        """Constructor for NeuralNet class.

        :param min_element: double, min element in random generation of weight matrixes
        :param max_element: double, max element in random generation of weight matrixes
        """
        # Присваивание значений, сброс к начальным условиям
        self.min_element = min_element  # Минимальное значение при случайной генерации матриц весов
        self.max_element = max_element  # Максимальное

        self.design = []  # Каждый элемент этого списка хранит в себе либо описание отдельного слоя
        # (количество нейронов, функция активации), либо подсети
        self.amount_of_neurons_in_layer = []
        self.activation_funcs = []
        self.design_len = 0  # Длина self.design
        self.tf_layers = []  # Графы вычислений для каждого слоя
        self.unroll_breaks = [0]  # Границы каждого слоя в развернутом векторе
        self.layers = []  # Здесь хранятся слои, как объекты типа Layer

        self.placeholders_dict = {}  # Словарь вида (tf.placeholder: numpy.array). Отвечает за матрицы весов.

        self.dim = 0  # Размерность развернутого вектора

        self.if_compile = False  # Была ли НС скомпилированна
        self.correctness = True  # Все ли корректно в параметрах нейроной сети
        self.x = None  # Placeholder для входных данных
        self.y = None  # Placeholder для обучающих выходных данных
        self.cost = None  # Переменная ценовой функции
        self.sess = None  # Сессия
        self.init = None  # Начальные значения
        self.output = None  # Переменная выхода НС
        self.train_type = ""  # Тип обучения
        self.amount_of_outputs = None  # Количество выходов
        self.output_activation_func = None  # Функция активации выходов

        self.cost_func = None  # Пользовательская ценовая функция
        self.optimize_params_dict = {}  # Параметры оптимизации

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------Методы, которые задают архитектуру НС-------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def add(self, amount_of_units, activation_func):
        """Adding full-connected (FC) layer

        :param amount_of_units: int, amount of neurons in layer
        :param activation_func: function, activation function fro this layer

        """
        current_layer = (amount_of_units, activation_func, 0)
        self.design.append(current_layer)

    def add_sub_nets(self, sub_nets, activation_funcs):
        """Adding sub nets

        :param sub_nets: iterable, list of sub nets
        :param activation_funcs:  iterable, list of functions, length must be same as amount of
            layers in every sub net

        """

        if self.is_correct(sub_nets):
            current_layer = (sub_nets, activation_funcs, 1)
            self.design.append(current_layer)
            self.correctness = True
        else:
            self.correctness = False

    @staticmethod
    def is_correct(sub_nets):
        """Verifies the teh sub nets parameters are correct: amount of layers in each sub net must be the same, if
        input (output) layer is in one sub net, it must be in each sub net.

        :param sub_nets: iterable, list of sub nets
        :return: bool, True if everything is correct, False otherwise

        """
        amount_of_layers = sub_nets[0].return_amount_of_layers()
        input_set = sub_nets[0].return_input_set()
        output_set = sub_nets[0].return_output_set()
        for sub_net in sub_nets:
            cur_amount_of_layers = sub_net.return_amount_of_layers()
            cur_input_set = sub_net.return_input_set()
            cur_output_set = sub_net.return_output_set()
            if amount_of_layers != cur_amount_of_layers:
                sys.stderr.write('Amount of layers must be same in all sub nets')
                return False

            if cur_input_set != input_set:
                sys.stderr.write('Input layer must be in all sub nets')
                return False

            if cur_output_set != output_set:
                sys.stderr.write('Output layer must be in all sub nets')
                return False
        return True

    def add_input_layer(self, amount_of_units):
        """Add input layer to neural network.

        :param amount_of_units: int, amount of neurons in input layer.
        """
        self.add(amount_of_units, None)

    def add_output_layer(self, amount_of_units, output_activation_func):
        """Add output layer to neural network

        :param amount_of_units: int, amount of neurons in output layer
        :param output_activation_func: function, activation func of output layer
        """
        self.amount_of_outputs = amount_of_units
        self.output_activation_func = output_activation_func
        self.add(self.amount_of_outputs, self.output_activation_func)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------Загрузка НС из файла-------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def load_net_from_file(self, filename):
        """Load neural net design from XML file.

        :param filename: string, path to XML file.
        """
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
        """Compile neural network (create tf graph corresponding to neural network). Without running this method a lot
        of methods can not working.

        """
        if not self.correctness:
            sys.stderr.write('Error in neural net design')
            return None
        self.sess = tf.Session()  # Создание сессии
        if self.if_compile:  # Проверка, была ли скомпилированна НС ранее...
            # ...если да - сброс к начальным параметрам
            self.layers = []
            self.tf_layers = []
            self.init = None
        # else:
        #    self.add(self.amount_of_outputs, self.output_activation_func)  # Добавление выходного слоя
        # Выходной слой добавляется здесь, так как необходиом гарантировать, что он является последним
        self.if_compile = True
        self.x = tf.placeholder(tf.double)  # Создание placeholder для входных данных...
        self.y = tf.placeholder(tf.double)  # ...и обучающих выходов

        for index, layer in enumerate(self.design):
            if layer[-1] == 0:
                self.amount_of_neurons_in_layer.append(layer[0])
                self.activation_funcs.append(layer[1])

            else:
                self.activation_funcs.extend(layer[1])
                sub_nets = layer[0]
                amount_of_layers = sub_nets[0].return_amount_of_layers()
                for i in range(amount_of_layers):
                    amount_of_neurons = 0
                    for sub_net in sub_nets:
                        amount_of_neurons += sub_net.return_amount_of_neurons(i)
                    self.amount_of_neurons_in_layer.append(amount_of_neurons)

        self._create_layers()
        self.init = tf.global_variables_initializer()  # Инициализатор переменных
        self.sess.run(self.init)  # Инициализация переменных
        for index, layer in enumerate(self.layers):
            if index == 0:
                current_layer = layer.activation_func(layer * self.x)
            else:
                prev_layer = self.tf_layers[index - 1]
                current_layer = layer.activation_func(layer * prev_layer)
            self.tf_layers.append(current_layer)
        self.output = self.tf_layers[-1]  # Выход нейронной сети - это последний слой => послдений элемент tf_layers
        self.dim = self.unroll_breaks[-1]

    def _create_layers(self):
        self.design_len = len(self.design)
        index = 0
        while index < len(self.design):
            layer = self.design[index]
            if layer[-1] == 0:
                self._add_fc_layer(index)
                index += 1
            else:
                amount_of_layers = self._add_sub_nets_layers(index)
                index += amount_of_layers

    def _add_sub_nets_layers(self, index):
        sub_nets = self.design[index][0]
        activation_funcs = self.design[index][1]
        amount_of_layers = sub_nets[0].return_amount_of_layers()
        for i in range(amount_of_layers - 1):
            cur_layer_size = []
            for sub_net in sub_nets:
                cur_layer_size.append(sub_net.return_layer_matrix_size(i))
            cur_layer = LayerBlocks(cur_layer_size, activation_funcs[i + 1])
            breaker = self.unroll_breaks[-1] + cur_layer.return_layer_dim()
            self.unroll_breaks.append(breaker)
            self.layers.append(cur_layer)
        if index != self.design_len - 1:
            current_layer_units = sub_nets[-1].return_amount_of_neurons(-1)
            next_layer_units = self.amount_of_neurons_in_layer[index + 1]
            activation_func = self.activation_funcs[index + 1]
            cur_layer = LayerFC((next_layer_units, current_layer_units), activation_func)
            breaker = self.unroll_breaks[-1] + cur_layer.return_layer_dim()
            self.unroll_breaks.append(breaker)
            self.layers.append(cur_layer)
        return amount_of_layers

    def _add_fc_layer(self, index):
        if index != self.design_len - 1:
            current_layer_units = self.amount_of_neurons_in_layer[index]
            next_layer_units = self.amount_of_neurons_in_layer[index + 1]
            activation_func = self.activation_funcs[index + 1]
            cur_layer = LayerFC((next_layer_units, current_layer_units), activation_func)
            breaker = self.unroll_breaks[-1] + cur_layer.return_layer_dim()
            self.unroll_breaks.append(breaker)
            self.layers.append(cur_layer)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------Методы, вычисляющие значение НС------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #

    def calc(self, calc_var, d):
        """Calculate any tf.Tensor which depends of neural network. Need a compiled NN.

        :param calc_var: tf.Tensor, which you need to calculate
        :param d: dictionary, input data for NN. In most cases d = {net.x: numpy.array([[...], [...],...])}
        :return: numpy.array([[...], [...], ...]), calculated value of tf.Tensor calc_var
        """
        d.update(self.placeholders_dict)  # Добавляем в словарь d placeholder для матриц весов
        return self.sess.run(calc_var, d)

    def run(self, inputs):
        """Calculate output on input data inputs

        :param inputs: numpy.array([[...], [...],... ])
        :return: numpy.array([[...], [...],...])
        """
        result = self.calc(self.output, {self.x: inputs})
        return result

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------Методы, возвращающие различные параметры НС--------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def return_graph(self):
        """Return tf.Tensor which respect to output layer

        :return: tf.Tensor
        """
        return self.output

    def return_session(self):
        """Return tf.Session

        :return: tf.Session
        """
        return self.sess

    def return_unroll_dim(self):
        """Return dim of unrolled vector = amount of weights of neural network.

        :return: int
        """
        return self.dim

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------Методы, проводящие манипуляции с матрицами весов------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #

    def _assign_matrixes(self, assign_list):
        self.placeholders_dict.update(assign_list)

    def roll_matrixes(self, unroll_vector):
        """Roll vector into weight matrixes

        :param unroll_vector: numpy.array with dimension = net.dim
        """

        if not self.if_compile:
            sys.stderr.write('Compile model before roll matrixes')
            return None
        if len(unroll_vector) != self.dim:
            sys.stderr.write('Error in dimension of unroll vector')
        assign_list = []
        for index, layer in enumerate(self.layers):
            left_layer_break = self.unroll_breaks[index]  # Левая граница матрицы весов = правая гранница
            # вектора сдвига предыдущего слоя
            right_layer_break = self.unroll_breaks[index + 1]  # Правая граница матрицы весов =
            # = левая граница вектора сдвига
            layer_unroll_vector = unroll_vector[left_layer_break:right_layer_break]
            assign_list.extend(layer.roll_matrix(layer_unroll_vector))
        self._assign_matrixes(assign_list)

    def set_random_matrixes(self):
        unroll_vector = numpy.random.uniform(self.min_element, self.max_element, self.dim)
        self.roll_matrixes(unroll_vector)

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------Прочее-------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    def init_params(self):
        # Инициализация начальных параметров
        self.sess.run(self.init)

    def set_cost_func(self, cost_func):
        self.cost_func = cost_func

    def user_cost(self, params):
        return self.cost_func(self, params)

    def optimize(self, params_dict, cost_func, end_cond, min_cost):
        self.set_cost_func(cost_func)
        dim = self.unroll_breaks[-1][-1]
        res = optimize(params_dict, self.user_cost, dim, end_cond, min_cost=min_cost)
        return res

    def save_weights(self, filename):
        matrixes = dict.values(self.placeholders_dict)
        unroll_vector = numpy.empty(0)
        for matrix in matrixes:
            cur_unroll_vector = matrix.reshape(matrix.size)
            unroll_vector = numpy.append(unroll_vector, cur_unroll_vector)
        numpy.save(filename, unroll_vector)

    def load_weights(self, filename):
        unroll_vector = numpy.load(filename + '.npy')
        self.roll_matrixes(unroll_vector)
