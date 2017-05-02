import numpy as np
import sys
import math

class LinearRegression(object):
  def __init__(self, alpha = 0.05, max_iter = 10000):
    self.theta_ = None
    self.alpha_ = alpha
    self.max_iter_ = max_iter

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    # Add a constant offset.
    X.resize( (X.shape[0], X.shape[1] + 1) )
    for i in range(len(X)):
      X[i][-1] = 1.0

    assert len(X.shape) == 2
    num_samples, num_dimensions = X.shape
    assert len(y.shape) == 1
    assert len(y) == num_samples

    self.theta_ = np.zeros(num_dimensions)

    prev_cost = sys.maxint
    for i in range(self.max_iter_):
      self.theta_ = self.theta_ - \
                    self.alpha_ * (1.0 / num_samples) \
                                * X.transpose()\
                                   .dot(X.dot(self.theta_) - y)
      curr_cost = self.compute_cost(X, y, self.theta_)
      print 'iter', i, ': cost =', curr_cost
      if math.fabs(curr_cost - prev_cost) < 1e-6:
        break
      prev_cost = curr_cost

  def predict(self, X):
    X = np.array(X)
    X.resize( (X.shape[0], X.shape[1] + 1) )
    for i in range(len(X)):
      X[i][-1] = 1.0

    # print self.theta_
    return X.dot(self.theta_)

  def compute_cost(self, X, y, theta):
    return 1.0 / (2 * y.shape[0]) * (X.dot(theta) - y).dot((X.dot(theta) - y))

if __name__ == '__main__':
  X = np.array([[1.0, 3.0, 4.0],\
                [2.0, 4.0, 5.0],\
                [3.0, 5.0, 6.0]])
  y = np.array([4.0, 5.0, 6.0])

  model = LinearRegression()
  model.fit(X, y)
  print model.predict(np.array([[1.0, 1.5, 1.8]]))