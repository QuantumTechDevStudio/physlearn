# Необходмые команды импорта.
import numpy as np
from numpy import linalg as LA
from numpy import linalg as LA
import numpy.random as rand
from scipy.integrate import quad
from scipy.integrate import trapz
from matplotlib import pylab as plt
from tqdm import tqdm_notebook
from IPython.display import clear_output
import tensorflow as tf
from physlearn.NeuralNet.NeuralNet import NeuralNet
from physlearn.Optimizer import Optimizer
from physlearn.Lagaris import ann_constructor
from physlearn.Lagaris import math_util


class LagarisSolver:
# Под psi подразумевается пробная функция exp(-beta*x^2)*ANN(weights, x)

	def __init__(self):
		self.a = None
		self.b = None
		self.train_x = None
		self.integrator = None
		self.trapz_x = None
		self.net = None
		self.din = None
		self.net_output = None
		self.sess = None
		self.psi = None
		self.H_psi = None
		self.cost_func = None
		self.beta = None


	def define_psi(self, sigmoid_ammount):
		self.net, self.net_output, net_sum, self.sess = ann_constructor.return_net_expressions(1, sigmoid_ammount)
		self.beta = tf.placeholder(tf.float64, shape = ())
		self.psi = tf.exp(-tf.abs(self.beta)*tf.square(self.net.x))*self.net_output
		self.dim = self.net.return_unroll_dim()


	def define_H_psi(self, V):
		first_deriative = tf.gradients(self.psi, self.net.x)[0]
		self.H_psi = (-(tf.gradients(first_deriative, self.net.x)[0]) + tf.multiply(V, self.psi))
		self.psi_psi = tf.square(self.psi)
		self.psi_H_psi = tf.multiply(self.psi, self.H_psi)  


	def calc_H_psi(self, x, beta_loc):
		x = np.reshape(x, (1, np.size(x)))
		return self.net.calc(self.H_psi, {self.net.x : x, self.beta : beta_loc})[0]


	def calc_psi(self, x, beta_loc):
		x = np.reshape(x, (1, np.size(x)))
		return self.net.calc(self.psi, {self.net.x : x, self.beta : beta_loc})[0]


	def psi_H_psi_integrand(self, x, beta_loc):
		x = np.reshape(x, (1, np.size(x)))
		return self.net.calc(self.psi_H_psi, {self.net.x : x, self.beta : beta_loc})
	

	def psi_psi_integrand(self, x, beta_loc):
		x = np.reshape(x, (1, np.size(x)))
		return self.net.calc(self.psi_psi, {self.net.x : x, self.beta : beta_loc})




	def integrate_quad(self, func, beta_loc):
		res, _ = quad(func, self.a, self.b, args=(beta_loc))
		return res

	def integrate_trapz(self, func, x, beta_loc):
		y = func(x, beta_loc)
		return trapz(y , x = x, axis = -1)



	def J(self, params):
		self.net.roll_matrixes(params[0:-1])
		beta_loc = params[-1]
		psi = self.calc_psi(self.train_x, beta_loc)
		H_psi = self.calc_H_psi(self.train_x, beta_loc)

		if (self.integrator == 'quad'):
			mean_E = self.integrate_quad(self.psi_H_psi_integrand, beta_loc)
			norm = self.integrate_quad(self.psi_psi_integrand, beta_loc)
		else:
			mean_E = self.integrate_trapz(self.psi_H_psi_integrand, self.trapz_x, beta_loc)
			norm = self.integrate_trapz(self.psi_psi_integrand, self.trapz_x, beta_loc)	

		eps = mean_E / norm

		cost = np.sum(np.square(H_psi - eps*psi)) / norm + (norm - 1)**2   
		return cost


	def get_cost_func_quad(self, train_x, a, b):
		self.a = a
		self.b = b
		self.integrator = 'quad'
		self.train_x = train_x
		return self.J


	def get_cost_func_trapz(self, train_x, trapz_x):
		self.a = trapz_x[0]
		self.b = trapz_x[-1]
		self.train_x = train_x
		self.integrator = 'trapz'
		self.trapz_x = trapz_x
		return self.J


	def get_psi(self):
		return self.psi


	def get_H_psi(self):
		return self.H_psi


	def get_net(self):
		return self.net


	def get_net_x(self):
		return self.net.x

	def get_dim(self):
		return self.dim

	def get_sess(self):
		return self.sess