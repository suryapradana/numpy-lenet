import numpy as np

class Softmax(object):
	def __init__(self, shape):
		self.softmax = np.zeros(shape)
		self.eta = np.zeros(shape)
		self.batch_size = shape[0]
	
	def predict(self, prediction):
		exp_predict = np.zeros(prediction.shape)
		self.softmax = np.zeros(prediction.shape)
		for i in range(self.batch_size):
			prediction[i, :] -= np.max(prediction[i, :])
			exp_predict[i] = np.exp(prediction[i])
			self.softmax[i] = exp_predict[i] / np.sum(exp_predict[i])
		return self.softmax

	def calc_loss(self, pred, label):
		self.label = label
		self.pred = pred
		self.predict(pred)
		self.loss = 0
		for i in range(self.batch_size):
			self.loss += np.log(np.sum(np.exp(pred[i]))) - pred[i, label[i]]
		return self.loss

	def gradient(self):
		self.eta = self.softmax.copy()
		for i in range(self.batch_size):
			self.eta[i, self.label[i]] -= 1
		return self.eta

