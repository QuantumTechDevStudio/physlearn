from tqdm import tqdm


class Pauell:
    func = None
    end_cond = None

    epsilon1 = None
    epsilon2 = None

    delta_x = None
    x = [0 for _ in range(3)]
    f = [0 for _ in range(3)]

    x_min = None
    f_min = None

    x_tilda = None
    f_tilda = None
    if_calc = False

    cost_list = []

    def __init__(self, delta_x, epsilon1, epsilon2):
        self.delta_x = delta_x
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    def optimize(self, func, end_cond, x0):
        self.func = func
        self.end_cond = end_cond
        self.x[0] = x0
        self.f[0] = func(self.x[0])
        for _ in range(end_cond):
            if not self.if_calc:
                self.calc_points()

            self.f_min = min(self.f)
            self.x_min = self.x[self.f.index(self.f_min)]

            up_part = ((self.x[1] ** 2) - (self.x[2] ** 2)) * self.f[0] + ((self.x[2] ** 2) - (self.x[0] ** 2)) \
                      * self.f[1] + ((self.x[0] ** 2) - (self.x[1] ** 2)) * self.f[2]

            down_part = (self.x[1] - self.x[2]) * self.f[0] + (self.x[2] - self.x[0]) * self.f[1] \
                        + (self.x[0] - self.x[1]) * self.f[2]

            self.x_tilda = 0.5 * up_part / down_part
            self.f_tilda = func(self.x_tilda)

            #print(self.f_min, self.f_tilda)
            check1 = abs(self.f_min - self.f_tilda)
            check2 = abs(self.x_min - self.x_tilda)
            #print(check1, check2)
            #print(check1<self.epsilon1, check2<self.epsilon2)
            #print('-----------------')

            if (check1 < self.epsilon1) and (check2 < self.epsilon2):
                #print('HUI')
                return self.x_min, self.f_min
            else:
                if (self.x[0] >= self.x_tilda) and (self.x[2] <= self.x_tilda):
                    if self.f_min <= self.f_tilda:
                        self.x[0] = self.x_min - self.delta_x
                        self.x[1] = self.x_min
                        self.x[2] = self.x_min + self.delta_x
                    else:
                        self.x[0] = self.x_tilda - self.delta_x
                        self.x[1] = self.x_tilda
                        self.x[2] = self.x_tilda + self.delta_x
                    self.if_calc = True
                else:
                    self.x[0] = self.x_tilda
                    self.if_calc = False
        #print('HUI2')
        return self.x_tilda, self.f_tilda

    def calc_points(self):
        self.x[1] = self.x[0] + self.delta_x
        self.f[1] = self.func(self.x[1])
        if self.f[0] > self.f[1]:
            self.x[2] = self.x[0] + 2 * self.delta_x
        else:
            self.x[2] = self.x[0] - self.delta_x
        self.f[2] = self.func(self.x[2])
