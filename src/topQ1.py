import sys

from gen import gen
import matplotlib.pyplot as plt

from classifier1Copy import classifier
import numpy as np

l = 10
m = 100
n = 1000
p = 50000

(Y,X) = gen(l, m, n, p, False)
#
c = classifier((n))
#print X.shape
#
temp = Y.tolist()
temp1 = X.tolist()
#print len(temp)
#print int(float(9)/10 * p)
X = np.array(temp1[0: int(float(1)/10 * p)])
#print X.shape
Y = np.array(temp[0: int(float(1)/10 * p)])
#X = np.array(temp1)
#Y = np.array(temp)
X_test = np.array(temp1[int(float(9)/10 * p): p])
Y_test = np.array(temp[int(float(9)/10 * p): p])
#temp = temp1 = None



#c.tune(X, Y, X_test, Y_test, PWMeta = [1.5, 0.25, 0.03, 0.005, 0.001], WWOalpha = [1.1, 1.01], WWMalpha = [1.1], WWMgamma = [2.0, 0.3, 0.04, 0.006, 0.001], ADAeta = [1.5, 0.25, 0.03, 0.005, 0.001], n = n)

(Y,X) = gen(l, m, n, p, False)
ret = c.tune(X, Y, X_test, Y_test, PWMeta = [ 0.03], WWOalpha = [1.1], WWMalpha = [1.1], WWMgamma = [2.0], ADAeta = [1.5], cycle = 1, n = n,flag = False)
#for j in range(20):
#    for i in range(int(float(1)/5 * p)):
#       c.train(X[i], Y[i])

N = range(50000)
plt.figure()
plt.title('n = %d' % n)
plt.plot(N, ret[0], c='r', label="Perceptron")
plt.plot(N, ret[1], c='g', label="Perceptron with margin")
plt.plot(N, ret[2], c='b', label="Winnow")
plt.plot(N, ret[3], c='y', label="Winnow with margin")
plt.plot(N, ret[4], c='k', label="AdaGrad")
plt.legend(loc=2)
plt.show()

#c.test(X_test, Y_test)


