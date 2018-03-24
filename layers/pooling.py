import numpy as np
import matplotlib.pyplot as plt
import cv2

class MaxPooling(object):
	def __init__(self, shape, ksize=2, stride=2):
		self.input_shape = shape
		self.ksize = ksize
		self.stride = stride
		self.output_channels = shape[-1]
		self.index = np.zeros(shape)
		self.output_shape = [shape[0], shape[1] / self.stride, shape[2] / self.stride, self.output_channels]

	def forward(self, x):
		out = np.zeros([x.shape[0], x.shape[1] / self.stride, x.shape[2] / self.stride, self.output_channels])
		for b in range(x.shape[0]):
			for c in range(self.output_channels):
				for i in range(0, x.shape[1], self.stride):
					for j in range(0, x.shape[2], self.stride):
						out[b, i / self.stride, j / self.stride, c] = np.max(x[b, i:i + self.ksize, j:j + self.ksize, c])
						index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
						self.index[b, i + index / self.stride, j + index % self.stride, c] = 1

		return out

	def gradient(self, eta):
		return np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2) * self.index

def test_max_pooling():
	im = cv2.imread('test.jpg')
	im2 = cv2.imread('test.jpg')
	im = np.array([im, im2]).reshape([2, im.shape[0], im.shape[1], im.shape[2]])

	pool = MaxPooling(im.shape, 2, 2)
	im1 = pool.forward(im)
	im2 = pool.gradient(im1)

	print('im: ', im[1, :, :, 1])
	print('im1: ', im1[1, :, :, 1])
	print('im2: ', im2[1, :, :, 1])

	plt.imshow(im1[0])
	plt.show()

if __name__ == '__main__':
	test_max_pooling()