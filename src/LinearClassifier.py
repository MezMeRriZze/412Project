import numpy as np
from abc import ABCMeta, abstractmethod

#cited from mengxiong liu
class LinearClassifier(object):
	def predict(self, X):
		return np.array([self.evaluate(self.weight, x) for x in X])

	def test(self, weight, X, Y):
		assert len(X) == len(Y)
		loss = 0.0
		for i in xrange(len(X)):
			if self.evaluate(weight, X[i]) != Y[i]:
				loss += 1.0
		return loss

	def evaluate(self, weight, x):
		return 1 if np.dot(weight, np.append(x, 1)) > 0 else -1

	def wrong(self):
		self.R = 0
		self.W += 1

	def right(self):
		self.R += 1

	def mistake(self, weight, x, y):
		return y * np.dot(weight, np.append(x, 1)) < 0




class Perceptron(LinearClassifier):
	def __init__(self, gamma=0):
		self.gamma = gamma

	def tune(self, X, Y, cycle=20, etas=[1], R=None):
		assert len(X) == len(Y)
		error = []
		for eta in etas:
			perm = np.random.permutation(range(len(X)))
			weight = self.fit(X[perm[: len(X) / 10]], Y[perm[: len(Y) / 10]], eta, cycle)
			error.append(self.test(weight, X[perm[len(X) / 10 : 2 * len(X) / 10]], Y[perm[len(X) / 10 : 2 * len(X) / 10]]))
		self.eta = etas[np.argmin(error)]
		self.weight = self.fit(X, Y, self.eta, cycle, R)
		print error
		print "eta: %f" % self.eta
		print "error rate: %f" % (self.test(self.weight, X, Y) / len(Y))

	def fit(self, X, Y, eta, cycle, R=None):
		assert len(X) == len(Y)
		if R != None:
			self.W, self.R = 0, 0
		weight = np.zeros(len(X[0]) + 1)
		bias = 0
		for c in xrange(cycle):
			for i in xrange(len(X)):
				if Y[i] * (np.dot(weight, np.append(X[i], 1))) > self.gamma:
					if R != None:
						self.right()
						if self.R == R:
							return weight
					continue
				if R != None:
					if self.mistake(weight, X[i], Y[i]):
						self.wrong()
				weight = self.gradient(weight, X[i], Y[i], eta)
		return weight

	def record(self, X, Y):
		assert len(X) == len(Y)
		W = []
		m = 0
		weight = np.zeros(len(X[0]) + 1)
		for i in xrange(len(X)):
			if Y[i] * (np.dot(weight, np.append(X[i], 1))) <= self.gamma:
				if self.mistake(weight, X[i], Y[i]):
					m += 1
				weight = self.gradient(weight, X[i], Y[i], self.eta)
			W.append(m)
		return W

	def gradient(self, weight, x, y, eta):
		return weight + eta * y * np.append(x, 1)



class Winnow(LinearClassifier):
	def __init__(self):
		pass

	def tune(self, X, Y, cycle=20, alphas=None, gammas=[0], R=None):
		assert len(X) == len(Y)
		error = []
		for alpha in alphas:
			for gamma in gammas:
				perm = np.random.permutation(range(len(X)))
				weight = self.fit(X[perm[: len(X) / 10]], Y[perm[: len(Y) / 10]], alpha, gamma, cycle)
				error.append(self.test(weight, X[perm[len(X) / 10 : 2 * len(X) / 10]], Y[perm[len(X) / 10 : 2 * len(X) / 10]]))
		self.alpha = alphas[np.argmin(error) / len(gammas)]
		self.gamma = gammas[np.argmin(error) % len(gammas)]
		self.weight = self.fit(X, Y, self.alpha, self.gamma, cycle, R)
		print error
		print "alpha: %f" % self.alpha
		print "gamma: %f" % self.gamma
		print "error rate: %f" % (self.test(self.weight, X, Y) / len(Y))

	def fit(self, X, Y, alpha, gamma, cycle, R=None):
		assert len(X) == len(Y)
		if R != None:
			self.W, self.R = 0, 0
		weight = np.ones(len(X[0]))
		bias = -len(X[0])
		for c in xrange(cycle):
			for i in xrange(len(X)):
				if Y[i] * (np.dot(weight, X[i]) + bias) > gamma:
					if R != None:
						self.right()
						if self.R == R:
							return np.append(weight, bias)
					continue
				if R != None:
					if self.mistake(np.append(weight, bias), X[i], Y[i]):
						self.wrong()
				weight = self.gradient(weight, bias, X[i], Y[i], alpha)
		return np.append(weight, bias)

	def record(self, X, Y):
		W = []
		m = 0
		weight = np.ones(len(X[0]))
		bias = -len(X[0])
		for i in xrange(len(X)):
			if Y[i] * (np.dot(weight, X[i]) + bias) <= self.gamma:
				if self.mistake(np.append(weight, bias), X[i], Y[i]):
					m += 1
				weight = self.gradient(weight, bias, X[i], Y[i], self.alpha)
			W.append(m)
		return W


	def gradient(self, weight, bias, x, y, alpha):
		return np.multiply(weight, np.power(alpha, x * y))


class AdaGrad(LinearClassifier):
	def __init__(self):
		pass

	def tune(self, X, Y, cycle=20, etas=None, R=None, report_loss=False):
		assert len(X) == len(Y)
		error = []
		for eta in etas:
			perm = np.random.permutation(range(len(X)))
			weight = self.fit(X[perm[: len(X) / 10]], Y[perm[: len(Y) / 10]], eta, cycle)
			error.append(self.test(weight, X[perm[len(X) / 10 : 2 * len(X) / 10]], Y[perm[len(X) / 10 : 2 * len(X) / 10]]))
		self.eta = etas[np.argmin(error)]
		self.weight = self.fit(X, Y, self.eta, cycle, R, report_loss)
		print error
		print "eta: %f" % self.eta
		print "error rate: %f" % (self.test(self.weight, X, Y) / len(Y))

	def fit(self, X, Y, eta, cycle, R=None, report_loss=False):
		assert len(X) == len(Y)
		if R != None:
			self.W, self.R = 0, 0
		if report_loss:
			self.misclassification, self.hinge = [], []
		weight = np.zeros(len(X[0]) + 1)
		G = np.ones(len(X[0]) + 1)
		for c in xrange(cycle):
			for i in xrange(len(X)):
				if Y[i] * (np.dot(weight, np.append(X[i], 1))) > 1:
					if R != None:
						self.right()
						if self.R == R:
							return weight
					continue
				if R != None:
					if self.mistake(weight, X[i], Y[i]):
						self.wrong()
				gradient = self.gradient(weight, X[i], Y[i])
				G += np.square(gradient)
				weight -= eta * np.divide(gradient, np.sqrt(G))
			if report_loss:
				self.misclassification.append(self.test(weight, X, Y))
				self.hinge.append(self.hinge_loss(weight, X, Y))
		return weight

	def record(self, X, Y):
		W = []
		m = 0
		weight = np.zeros(len(X[0]) + 1)
		G = np.ones(len(X[0]) + 1)
		for i in xrange(len(X)):
			if Y[i] * (np.dot(weight, np.append(X[i], 1))) <= 1:
				if self.mistake(weight, X[i], Y[i]):
					m += 1
				gradient = self.gradient(weight, X[i], Y[i])
				G += np.square(gradient)
				weight -= self.eta * np.divide(gradient, np.sqrt(G))
			W.append(m)
		return W


	def gradient(self, weight, x, y):
		return -y * np.append(x, 1)

	def hinge_loss(self, weight, X, Y):
		assert len(X) == len(Y)
		return np.sum(np.array([max(0.0, 1.0 - Y[i] * self.evaluate(weight, X[i])) for i in xrange(len(X))]))
		

def main():
    


main()
