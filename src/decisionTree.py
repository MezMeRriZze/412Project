from sklearn import tree
from copy import deepcopy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from preprocess import *
import numpy as np

def cross_validate(classifier, X, Y, tree_name):
	assert len(X) == len(Y)
	accuracy = []
	dts = []
        opt_idx = -1
	for i in xrange(len(X)):
		train_X, train_Y = [], []
		test_X, test_Y = None, None
		for j in xrange(len(Y)):
			if i == j:
				test_X, test_Y = X[j], Y[j]
			else:
				train_X.append(X[j])
				train_Y.append(Y[j])
		train_X, train_Y = np.concatenate(train_X, axis=0), np.concatenate(train_Y, axis=0)
		classifier.fit(train_X, train_Y)
		accuracy.append(accuracy_score(test_Y, classifier.predict(test_X)))

		if tree_name is not None:
			dts.append(deepcopy(classifier))

	if tree_name is not None:
		opt_idx = np.argmax(accuracy)
                print "max : " + str(accuracy[opt_idx])
#		tree.export_graphviz(dts[opt_idx], out_file=tree_name)     
	mean, std = np.mean(accuracy), np.std(accuracy)
        print "mean : " + str(mean)
        print "std : " + str(std)
	return dts[opt_idx]

def trainTree(user, movie, train, maxDep = 16):
    X,Y = get_n_fold(user, movie, train, num_folds = 5)
    dt = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = maxDep)
    dt = cross_validate(dt, X, Y, "dt")
    return dt

def predict(user, movie, test, dt):
    ret = []
    X = get_test(user, movie, test)
    for item in X:
        out = dt.predict(np.array(item[1]))
        ret.append([item[0], out[0]])
    return ret

def main():
    user = readUser()
    movie = readMovie()
    print "length of user and movie"
    print len(user), len(movie)
    train = readTrain()
    test = readTest()
    dt = trainTree(user, movie, train, maxDep = None)
    res = predict(user, movie, test, dt)
    with open("../out/output.txt", "w") as f :
        f.write("Id,rating\n")
        for item in res :
            f.write(str(item[0]) + ", " + str(item[1]) + "\n")



main()
