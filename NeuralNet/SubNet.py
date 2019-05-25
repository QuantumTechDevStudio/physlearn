class SubNet:
    def __init__(self):
        self.design = []
        self.input_layer = ()
        self.output_layer = ()
        self.amount_of_layers = None
        self.output_activation_func = None
        self.input_set = False
        self.output_set = False
        self.amount_of_layers = 0

    def add(self, amount_of_units):
        current_layer = amount_of_units
        self.amount_of_layers += 1
        self.design.append(current_layer)

    def add_input_layer(self, amount_of_units):
        self.add(amount_of_units)
        self.input_set = True

    def add_output_layer(self, amount_of_units):
        self.add(amount_of_units)
        self.output_set = True

    def return_sizes(self):
        sizes_list = []
        for index in range(self.amount_of_layers - 1):
            weight = (self.design[index + 1], self.design[index])
            sizes_list.append(weight)
        return sizes_list

    def return_layer_matrix_size(self, layer_index):
        return self.design[layer_index + 1], self.design[layer_index]

    def return_amount_of_layers(self):
        return self.amount_of_layers

    def return_amount_of_neurons(self, layer_index):
        return self.design[layer_index]

    def return_output_set(self):
        return self.output_set

    def return_input_set(self):
        return self.input_set
