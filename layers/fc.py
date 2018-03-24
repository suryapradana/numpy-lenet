import numpy as np
from functools import reduce
import math


class FullyConnect(object):
	def __init__(self, shape, output_num=2):
		self.input_shape = shape
		self.batch_size = shape[0]

		input_len = reduce(lambda x, y: x * y, shape[1:])
		self.weights = np.random.standard_normal((input_len, output_num)) / 100
		self.bias = np.random.standard_normal(output_num) / 100

		self.output_shape = [self.batch_size, output_num]

		self.w_grad = np.zeros(self.weights.shape)
		self.b_grad = np.zeros(self.bias.shape)

	def forward(self, x):
		self.x = x.reshape([self.batch_size, -1])
		output = np.dot(self.x, self.weights) + self.bias
		return output

	def gradient(self, eta):
		for i in range(eta.shape[0]):
			col_x = self.x[i][:, np.newaxis]
			eta_i = eta[i][:, np.newaxis].T
			self.w_grad += np.dot(col_x, eta_i)
			self.b_grad += eta_i.reshape(self.bias.shape)

		next_eta = np.dot(eta, self.weights.T)
		next_eta = np.reshape(next_eta, self.input_shape)
		return next_eta

	def backward(self, lr=0.00001, weight_decay=0.0004):
		self.weights *= (1 - weight_decay)
		self.bias *= (1 - weight_decay)
		self.weights -= lr * self.w_grad
		self.bias -= lr * self.b_grad

		self.w_grad = np.zeros(self.w_grad.shape)
		self.b_grad = np.zeros(self.b_grad.shape)


def test_fc():
	img = np.array([[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1]])
	fc = FullyConnect(img.shape, 2)
	output = fc.forward(img)
	fc.gradient(np.array([[1,-2],[3,4]]))
	print('w_grad: ', fc.w_grad)
	print('b_grad: ', fc.b_grad)
	fc.backward()
	print('fc.weights: ', fc.weights)

if __name__ == '__main__':
	test_fc()