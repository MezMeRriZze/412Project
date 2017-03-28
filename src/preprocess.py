import numpy as np
from math import *

def readUser(numberOfIntervals = 5, maxAge = 60, maxOccupation = 21):
    ret = {}
    interval = float(maxAge) / numberOfIntervals
    with open("../files/user.txt", "r") as f:
        for l in f :
            app = [0 for i in range(2 +  numberOfIntervals + maxOccupation)]
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            if l[1] == "M" : 
                app[0] = 1
            elif l[1] == "F" :
                app[1] = 1
            try :
                age = int(l[2])
                app[2 + int(age / interval)] = 1 
            except : age = -1
            
            try :
                occ = int(l[3])
                app[2 + numberOfInterVals + occ] = 1
            except : occ = -1
            ret[int(l[0])] = app
    return ret

def readMovie(numberOfIntervals = 5, minAge = 1919, maxAge = 2000):
    ret = {}
    interval = float(maxAge - minAge) / numberOfIntervals
    gen = []
    with open("../files/movie.txt", "r") as f:
        for l in f :
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            if l[2] != "N/A":
                temp = l[2].split('|')
                for item in temp:
                    gen.append(item)    
    gen = list(set(gen))
    maxGen = len(gen)
    with open("../files/movie.txt", "r") as f:
        for l in f :
            app = [0 for i in range(numberOfIntervals + maxGen)]
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            try : 
                yr = int(l[1])
                app[(yr - minAge) / interval] = 1
            except : yr = 0
            if l[2] != "N/A":
                genre = l[2].split('|')
                for item in genre:
                    app[numberOfIntervals + gen.index(item)] = 1
            ret[int(l[0])] = app
    return ret

def readTrain():
    ret = []
    with open("../files/train.txt", "r") as f:
        for l in f :
            app = [0 for i in range(3)]
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            app[0] = int(l[1])
            app[1] = int(l[2])
            app[2] = int(l[3])
            ret.append(app)
    return ret

def readTest():
    ret = []
    with open("../files/test.txt", "r") as f:
        for l in f :
            app = [0 for i in range(3)]
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            app[0] = int(l[0])
            app[1] = int(l[1])
            app[2] = int(l[2])
            ret.append(app)
    return ret

def split_data(X, num_folds = 5):
    k = len(X) / num_folds
    ret = []
    for i in range(num_folds):
        app = []
        if i != num_folds - 1:
            for j in range(i * k, (i + 1) * k):
                app.append(X[j])
            ret.append(app)
        else :
            for j in range(i * k, len(X)):
                app.append(X[j])
            ret.append(app)
    return ret


def get_n_fold(user, movie, train, num_folds = 5):
    X = []
    Y = []
    for item in train:
        app = user[item[0]] + movie[item[1]]
        X.append(app)
        Y.append([item[2]])
    X = split_data(X)
    Y = split_data(Y)
    X = np.array(X)
    print X.shape
    Y = np.array(Y)
    print Y.shape
    return X,Y




