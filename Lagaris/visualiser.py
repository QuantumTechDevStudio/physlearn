import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import d1_osc
import ann_constructor


# Класс упрощает визуализацию для ИНС и образов.
class Visualiser(object):

    def __init__(self, solver):
        self.net = solver.get_net()
        self.trial_func = solver.get_trial_func()
        self.func_sum = tf.reduce_sum(input_tensor=self.trial_func, axis=0)
        self.func_images = solver.get_images()
        self.func_images_sum = tf.reduce_sum(input_tensor = self.func_images, axis = 0)
        self.M = solver.M


    def show_func_sum(self, x):
        y = self.__func_sum_value(x)
        plt.title('Output:')
        plt.grid(True)
        plt.plot(x[0], y)

    def show_funcs(self, x):
        y = self.__funcs_value(x)
        plt.title('Outputs: ')
        plt.grid(True)
        for i in range(self.M):
            func_i = y[i,:]
            plt.plot(x[0], func_i)

    def show_images(self, x):
        y = self.__images_value(x)
        plt.title('Images:')
        plt.grid(True)
        for i in range(M):
            func_image_i = y[i, :]
            plt.plot(x[0], func_image_i)

    def show_images_sum(self, x):
        y = self.__images_sum_value(x)
        plt.title('Output:')
        plt.grid(True)
        plt.plot(x[0], y)

    def plot_four(self, x):
        y1 = self.__funcs_value(x)
        y2 = self.__images_value(x)
        y3 = self.__func_sum_value(x)
        y4 = self.__images_sum_value(x)

        fig = plt.figure()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        ax1 = plt.subplot(221)
        ax1.set_title('Sum of original net outputs:')
        ax1.plot(x[0], y3, 'x')

        ax2 = plt.subplot(222)
        ax2.set_title('Sum of net images:')
        ax2.plot(x[0], y4, 'x')

        ax3 = plt.subplot(223)
        ax3.set_title('Original net outputs:')
        for i in range(self.M):
            func_i = y1[i,:]
            ax3.plot(x[0], func_i)

        ax4 = plt.subplot(224)
        ax4.set_title('Image of net outputs:')
        for i in range(self.M):
            image_i = y2[i,:]
            ax4.plot(x[0], image_i)
        fig.subplots_adjust(left=0, bottom=0, right=2, top=2, hspace=0.2, wspace=0.2)

    def __funcs_value(self, x):
        return self.net.calc(self.trial_func, {self.net.x : x})

    def __func_sum_value(self, x):
        return self.net.calc(self.func_sum, {self.net.x : x})

    def __images_value(self, x):
        return self.net.calc(self.func_images, {self.net.x: x})

    def __images_sum_value(self, x):
        return self.net.calc(self.func_images_sum, {self.net.x: x})

    @staticmethod
    def show_wf(n, x):
        plt.title('Wave function:')
        plt.grid(True)
        plt.plot(x[0], d1_osc.wf(n,x), "r--")

    @staticmethod
    def show_wf_system(n_max, x):
        plt.title('Wave functions system:')
        plt.grid(True)
        for i in range(n_max+1):
            plt.plot(x[0], d1_osc.wf(i,x))





