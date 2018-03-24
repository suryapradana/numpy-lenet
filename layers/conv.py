import numpy as np
from functools import reduce
import math

def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


class Conv2D(object):
	def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
		super(Conv2D, self).__init__()
		self.input_shape = shape
		self.output_channels = output_channels
		self.input_channels = shape[-1]
		self.batch_size = shape[0]
		self.stride = stride
		self.ksize = ksize
		self.method = method

		weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
		self.weights = np.random.standard_normal((ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
		self.bias = np.random.standard_normal(self.output_channels) / weights_scale

		if method == 'VALID':
			self.eta = np.zeros((shape[0], (shape[1] - ksize + 1) / self.stride, (shape[2] - ksize + 1) / self.stride, self.output_channels))
		elif method == 'SAME':
			self.eta = np.zeros((shape[0], shape[1] / self.stride, shape[2] / self.stride, self.output_channels))

		self.w_grad = np.zeros(self.weights.shape)
		self.b_grad = np.zeros(self.bias.shape)
		self.output_shape = self.eta.shape

		if (shape[1] - ksize) % stride != 0:
			print('input tensor height can not fit stride')
		if (shape[2] - ksize) % stride != 0:
			print('input tensor width can not fit stride')

	def forward(self, x):
		col_weights = self.weights.reshape([-1, self.output_channels])
		if self.method == 'SAME':
			x = np.pad(x, 
				((0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)), 
				'constant', constant_value=0)
		self.col_image = []
		conv_out = np.zeros(self.eta.shape)
		for i in range(self.batch_size):
			img_i = x[i][np.newaxis, :]
			self.col_image_i = im2col(img_i, self.ksize, self.stride)
			conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
			self.col_image.append(self.col_image_i)
		self.col_image = np.array(self.col_image)
		return conv_out

	def gradient(self, eta):
		self.eta = eta
		col_eta = np.reshape(eta, [self.batch_size, -1, self.output_channels])
		
		for i in range(self.batch_size):
			self.w_grad += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
		self.b_grad += np.sum(col_eta, axis=(0, 1))

		if self.method == 'VALID':
			pad_eta = np.pad(self.eta, ((0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
							'constant', constant_values=0)
		if self.method == 'SAME':
			pad_eta = np.pad(self.eta, ((0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
							'constant', constant_values=0)

		col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batch_size)])
		flip_weights = np.flipud(np.fliplr(self.weights))
		col_flip_weights = flip_weights.reshape([-1, self.input_channels])
		next_eta = np.dot(col_pad_eta, col_flip_weights)
		next_eta = np.reshape(next_eta, self.input_shape)
		return next_eta

	def backward(self, lr=0.00001, weight_decay=0.0004):
		self.weights *= (1 - weight_decay)
		self.bias *= (1 - weight_decay)
		self.weights -= lr * self.w_grad
		self.bias -= lr * self.b_grad

		self.w_grad = np.zeros(self.w_grad.shape)
		self.b_grad = np.zeros(self.b_grad.shape)

def test_conv():
	img = np.ones((1, 32, 32, 3))
	img *= 2
	conv = Conv2D(img.shape, 12, 3, 1)
	next_ = conv.forward(img)
	next1 = next_.copy() + 1
	conv.gradient(next1 - next_)
	print('conv.w_grad: ', conv.w_grad)
	print('conv.b_grad: ', conv.b_grad)
	conv.backward()


if __name__ == '__main__':
	test_conv()
