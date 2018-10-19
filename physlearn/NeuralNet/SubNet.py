class SubNet:
    design = []
    input_layer = ()
    output_layer = ()
    amount_of_outputs = None
    output_activation_func = None
    input_set = False
    output_set = False
    amount_of_layers = 0

    def add(self, amount_of_units, activation_func):
        current_layer = (amount_of_units, activation_func)
        self.amount_of_layers += 1
        self.design.append(current_layer)

    def add_input_layer(self, amount_of_units):
        # Добавление входного слоя
        # Функция активации входного слоя нигде не используется, но указывается self.linear,
        # как наследие прошлой реализации, хотя можно использовать и None
        self.add(amount_of_units, None)
        self.input_set = True

    def add_output_layer(self, amount_of_units, output_activation_func):
        self.add(amount_of_units, output_activation_func)
        self.output_set = True
        self.amount_of_layers += 1

    def return_sizes(self):
        sizes_list = []
        for index in range(self.amount_of_layers - 1):
            weight = (self.design[index + 1][0], self.design[index][0])
            sizes_list.append(weight)
        return sizes_list
