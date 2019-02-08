import sys

import tensorflow as tf

from physlearn.NeuralNet.Layer.LayerBlocks import LayerBlocks
from physlearn.NeuralNet.Layer.LayerFC import LayerFC
from physlearn.Optimizer.Optimizer import optimize


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

        self.dim = 0  # Размерность развернутого вектора
        self.cur_net_num = 0  # Номер создаваемой независимой НС

        self.if_compile = False  # Была ли НС скомпилированна
        self.layers_created = False
        self.correctness = True  # Все ли корректно в параметрах нейроной сети
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

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------Методы, которые задают архитектуру НС-------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def add(self, amount_of_units, activation_func):
        current_layer = (amount_of_units, activation_func, 0)
        self.design.append(current_layer)

    def add_sub_nets(self, sub_nets, activation_funcs):
        if self.is_correct(sub_nets):
            current_layer = (sub_nets, activation_funcs, 1)
            self.design.append(current_layer)
            self.correctness = True
        else:
            self.correctness = False

    @staticmethod
    def is_correct(sub_nets):
        amount_of_layers = sub_nets[0].return_amount_of_layers()
        input_set = sub_nets[0].return_input_set()
        output_set = sub_nets[0].return_output_set()
        for sub_net in sub_nets:
            cur_amount_of_layers = sub_net.return_amount_of_layers()
            cur_input_set = sub_net.return_input_set()
            cur_output_set = sub_net.return_output_set()
            if amount_of_layers != cur_amount_of_layers:  # Проверка на совпадение количества слоев во всех подсетях
                sys.stderr.write('Amount of layers must be same in all sub nets')
                return False

            if cur_input_set != input_set:  # Если в одной сети есть входной слой - он должен быть и во всех
                sys.stderr.write('Input layer must be in all sub nets')
                return False

            if cur_output_set != output_set:  # Аналогично с выходным
                sys.stderr.write('Output layer must be in all sub nets')
                return False
        return True

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
            self.tf_layers = []
            self.init = None

        self.if_compile = True
        self.x = tf.placeholder(tf.double)  # Создание placeholder для входных данных...

        for index, layer in enumerate(self.design):  # Проходимся по всем слоям
            if layer[-1] == 0:  # Если слой обычный полоносвязный
                self.amount_of_neurons_in_layer.append(layer[0])
                self.activation_funcs.append(layer[1])

            else:  # Если слой есть чать подсети
                self.activation_funcs.extend(layer[1])
                sub_nets = layer[0]
                amount_of_layers = sub_nets[0].return_amount_of_layers()
                for i in range(amount_of_layers):  # Проходимся по слоям
                    amount_of_neurons = 0  # Количество нейронов в слое
                    for sub_net in sub_nets:  # Проходимся по подсетям
                        amount_of_neurons += sub_net.return_amount_of_neurons(i)  # Суммируем число нейронов в i-ом слое
                        # в каждой подсети
                    self.amount_of_neurons_in_layer.append(amount_of_neurons)

        self.create_net()
        self.init = tf.global_variables_initializer()  # Инициализатор переменных
        self.sess.run(self.init)  # Инициализация переменных
        self.output = self.outputs[0]

    def _create_layers(self, unroll_vectors=None):
        self.cur_net_layers = []  # Список, в котором хранятся слои данной независимой НС
        if not self.layers_created:  # Если это первое создание НС...
            self.__create_new_layers()  # ...создаем новые слои
            self.layers_created = True
        else:
            self.__create_layers_like_created(unroll_vectors)  # Иначе создаем слои по типу уже созданных
        self.layers.append(self.cur_net_layers)

    def __create_new_layers(self):
        self.design_len = len(self.design)
        index = 0
        while index < len(self.design):
            layer = self.design[index]
            if layer[-1] == 0:
                self._add_fc_layer(index)  # Добавляем обычный плоносвязанный слой
                index += 1
            else:
                amount_of_layers = self._add_sub_nets_layers(index)  # Добавляем подсети
                index += amount_of_layers  # Увеличиваем значение счетчика на величину количества слоев в подсетях

    def _add_fc_layer(self, index):
        if index != self.design_len - 1:  # Проверка на то, что мы не в выходном слое
            current_layer_units = self.amount_of_neurons_in_layer[index]
            next_layer_units = self.amount_of_neurons_in_layer[index + 1]
            activation_func = self.activation_funcs[index + 1]
            cur_layer = LayerFC((next_layer_units, current_layer_units), activation_func, index)  # Создаем
            # полносвязанный слой
            # unroll_vector - это вектор, который представляет из себя результат "разворачивания" всех весовых матриц и
            # векторов сдвига в один большой вектор
            # i-ый элемент списка unroll_breaks содержит в себе индекс j: unroll_vector[j] - это последний элемент в
            # векторе unroll_vector, отвечающий слою i.
            breaker = self.unroll_breaks[-1] + cur_layer.return_layer_dim()
            self.unroll_breaks.append(breaker)
            self.cur_net_layers.append(cur_layer)

    def _add_sub_nets_layers(self, index):
        sub_nets = self.design[index][0]
        activation_funcs = self.design[index][1]
        amount_of_layers = sub_nets[0].return_amount_of_layers()
        for i in range(amount_of_layers - 1):  # Проходимся по всем слоям подсети, за исключением последнего (т.к он
            # должен быть полносвязным)
            cur_layer_size = []  # Список, в котором хранятся размеры весовых матриц каждой из подсетей, отвечающих
            # нужному слою
            for sub_net in sub_nets:
                cur_layer_size.append(sub_net.return_layer_matrix_size(i))
            cur_layer = LayerBlocks(cur_layer_size, activation_funcs[i + 1], index)
            breaker = self.unroll_breaks[-1] + cur_layer.return_layer_dim()
            self.unroll_breaks.append(breaker)
            self.cur_net_layers.append(cur_layer)
        if index != self.design_len - 1:  # Обрабатываем последний слой подсетей (если он не выходной), как обычный FC
            current_layer_units = sub_nets[-1].return_amount_of_neurons(-1)
            next_layer_units = self.amount_of_neurons_in_layer[index + 1]
            activation_func = self.activation_funcs[index + 1]
            cur_layer = LayerFC((next_layer_units, current_layer_units), activation_func, index)
            breaker = self.unroll_breaks[-1] + cur_layer.return_layer_dim()
            self.unroll_breaks.append(breaker)
            self.cur_net_layers.append(cur_layer)
        return amount_of_layers

    def __create_layers_like_created(self, unroll_vectors=None):
        # Если уже один раз создали сеть, то новые создаем по образцу и подобию
        # unroll_vectors - список, в котором хранятся развернутые вектора матриц для каждого слоя
        created_layers = self.layers[0]  # Список, где хранятся все слои уже созданной НС
        for index, layer in enumerate(created_layers):
            cur_layer_vector = unroll_vectors[index]
            layer_type = layer.return_class_name()  # Определяем тип слоя
            new_layer = None
            # В зависимости от типа создаем слой нужного вида
            if layer_type == 'LayerFC':
                new_layer = LayerFC(index=index)
            elif layer_type == 'LayerBlocks':
                new_layer = LayerBlocks(index=index)
            # Задаем параметры слоя
            new_layer.set_params(layer.weight_matrix, layer.weight_matrix_shape, layer.weight_dim, layer.bias_vector,
                                 layer.bias_dim, layer.activation_func, layer.dim)
            new_layer.create_layer_from_params(cur_layer_vector)  # Окончательно создаем слой с нужной матрицей
            self.cur_net_layers.append(new_layer)

    def create_net(self, unroll_vectors=None):
        # Метод создает независимую НС
        self._create_layers(unroll_vectors)  # Создаем слои

        for index, layer in enumerate(self.layers[self.cur_net_num]):  # Проходимся по слоям
            if index == 0:
                current_layer = layer.activation_func(layer * self.x)
            else:
                prev_layer = self.tf_layers[index - 1]
                current_layer = layer.activation_func(layer * prev_layer)
            self.tf_layers.append(current_layer)
        cur_output = self.tf_layers[-1]  # Выход нейронной сети - это последний слой => послдений элемент tf_layers
        self.outputs.append(cur_output)
        self.dim = self.unroll_breaks[-1]
        self.cur_net_num += 1
    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------Методы, вычисляющие значение НС------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #

    def calc(self, calc_var, x):
        # Метод вычисляет значение любого тензора, так или иначе связанного с выходным тензором НС
        # x - значение placeholder self.x
        return self.sess.run(calc_var, {self.x: x})

    def run(self, inputs):
        # Метод вычисляет выход НС на входных данных inputs
        result = self.calc(self.output, inputs)
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

    def roll_matrixes(self, unroll_vector, train_mode=False):
        # Метод "сворачивает" unroll_vector обратно в матрицы и подставляет их в качестве матриц весов
        # Метод имеет два режима работы
        # train_mode=False - по матрицам, полученным из unroll_vector строится новая НС, отвечающиая этим матрицам.
        # Затем эта нейросеть заменяет собой предыдущую. В этом режиме метод ничего не возвращает
        # train_mode=True - метод строит НС по матрицам, получаемым из unroll_vector, после чего возвращает выходной
        # тензор полученной НС, а все остальное удаляется
        if not self.if_compile:
            sys.stderr.write('Compile model before roll matrixes')
            return None

        unroll_vectors = []
        for index, _ in enumerate(self.layers[0]):
            left_layer_break = self.unroll_breaks[index]  # Левая граница матрицы весов = правая гранница
            # вектора сдвига предыдущего слоя
            right_layer_break = self.unroll_breaks[index + 1]  # Правая граница матрицы весов =
            # = левая граница вектора сдвига
            layer_unroll_vector = unroll_vector[left_layer_break:right_layer_break]
            unroll_vectors.append(layer_unroll_vector)
        self.create_net(unroll_vectors)

        if train_mode:
            output_tensor = self.outputs[-1]
            self.outputs.pop()
            self.layers.pop()
            self.cur_net_num -= 1
            return output_tensor
        else:
            self.outputs[0] = self.outputs[-1]
            self.layers[0] = self.layers[-1]
            self.output = self.outputs[-1]
            self.outputs.pop()
            self.layers.pop()
            self.cur_net_num -= 1

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
        dim = self.unroll_breaks[-1]
        res = optimize(params_dict, self.user_cost, dim, end_cond, min_cost=min_cost)
        return res
