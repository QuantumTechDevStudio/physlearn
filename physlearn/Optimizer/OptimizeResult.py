class OptimizeResult:
    cost_list = []

    def __init__(self, is_converge, amount_of_iterations, total_time, cost_list, exit_code, reason_of_exit, x):
        self.is_converge = is_converge
        self.amount_of_iterations = amount_of_iterations
        self.cost_function = cost_list[-1]
        self.reason_of_exit = reason_of_exit
        self.cost_list = cost_list
        self.exit_code = exit_code
        self.total_time = total_time
        self.x = x

    def __repr__(self):
        is_converge_str = "Is converge: " + str(self.is_converge) + '\n'
        amount_of_iterations_str = "Amount of iterations: " + str(self.amount_of_iterations) + '\n'
        total_time_str = "Total time: " + '{:.2f}'.format(self.total_time) + ' s\n'
        cost_function_str = "Reached function value: " + str(self.cost_function) + '\n'
        reason_of_exit_str = "Reason of break: " + str(self.reason_of_exit) + '\n'
        result = is_converge_str + amount_of_iterations_str + total_time_str + cost_function_str + reason_of_exit_str
        return result

    def __str__(self):
        is_converge_str = "Is converge: " + str(self.is_converge) + '\n'
        amount_of_iterations_str = "Amount of iterations: " + str(self.amount_of_iterations) + '\n'
        total_time_str = "Total time: " + '{:.2f}'.format(self.total_time) + ' s\n'
        cost_function_str = "Reached function value: " + str(self.cost_function) + '\n'
        reason_of_exit_str = "Reason of break: " + str(self.reason_of_exit) + '\n'
        result = is_converge_str + amount_of_iterations_str + total_time_str + cost_function_str + reason_of_exit_str
        return result
