import numpy as np
from math import *

def readUser(numberOfIntervals = 5, maxAge = 60, maxOccupation = 21):
    ret = {}
    interval = float(maxAge + 1) / numberOfIntervals
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

def prepareUserLookUp(numberOfIntervalsUser = 5, maxOccupation = 21):
    ptr = 0
    userLookUp = {}
    for gender in ["M", "F", "N/A"]:
        for i in ([("%03d" % j) for j in range(numberOfIntervalsUser)] + ["N/A"]):
            for occ in ([("%03d" % o) for o in range(maxOccupation)] + ["N/A"]):
                userLookUp[gender + " " + i + " " + occ] = ptr
                ptr += 1
    return userLookUp

def prepareMovieLookUp(numberOfIntervalsMovie = 5):
    genCombs = []
    with open("../files/movie.txt", "r") as f:
        for l in f :
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            temp = l[2].split('|')
            temp = sorted(temp)
            genCombs.append(' '.join(temp))
    genCombs= sorted(list(set(genCombs)))
    movieLookUp = {}
    ptr = 0
    for i in ([("%03d" % i) for i in range(numberOfIntervalsMovie)] + ["N/A"]):
        for item in genCombs :
            movieLookUp[i +" " + item] = ptr
            ptr += 1
    return movieLookUp

def buildMovieTable(movieLookUp, numberOfIntervalsMovie = 5, minAge=1919, maxAge = 2000):
    interval = float(maxAge + 1 - minAge) / numberOfIntervalsMovie
    movie = {}
    with open("../files/movie.txt", "r") as f:
        for l in f :
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            try : 
                yr = int(l[1])
                yr = "%03d" % ((yr - minAge) / interval)
            except : yr = "N/A"
            genre = l[2].split('|')
            genre = sorted(genre)
            genre = ' '.join(genre)
            movie[int(l[0])] = movieLookUp[yr + " " + genre]
    return movie

def buildUserTable(userLookUp, numberOfIntervalsUser = 5, maxOccupation = 21, maxAge= 60):
    interval = float(maxAge + 1) / numberOfIntervalsUser
    user = {}
    with open("../files/user.txt", "r") as f:
        for l in f:
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            gender = l[1]
            try :
                age = int(l[2])
                age = "%03d" % (int(age / interval))
            except : age = "N/A"
            try :
                occ = int(l[3])
                occ = "%03d" % occ
            except : occ = "N/A"
            user[int(l[0])] = userLookUp[gender + " " + age + " " + occ]
    return user
            
def prepareSpecialSVD(numberOfIntervalsUser = 5, numberOfIntervalsMovie = 5, maxOccupation = 21, minAge=  1919, maxAge = 2000, maxAgeUser = 60):
    userLookUp = prepareUserLookUp(numberOfIntervalsUser = numberOfIntervalsUser, maxOccupation = maxOccupation)
    movieLookUp = prepareMovieLookUp(numberOfIntervalsMovie =  numberOfIntervalsMovie)
    movie = buildMovieTable(movieLookUp, numberOfIntervalsMovie = numberOfIntervalsMovie, minAge= minAge, maxAge= maxAge)
    user = buildUserTable(userLookUp, numberOfIntervalsUser = numberOfIntervalsUser, maxOccupation = maxOccupation, maxAge= maxAgeUser)
    ret = [[[0,0.0001] for i in range(len(movieLookUp))] for j in range(len(userLookUp))]
    train = readTrain()
    for item in train:
        u = item[0]
        m = item[1]
        r = item[2]
        ret[user[u]][movie[m]][0] += r
        ret[user[u]][movie[m]][1] += 1
    temp = np.zeros((len(userLookUp), len(movieLookUp)))
    for i in range(len(userLookUp)):
        for j in range(len(movieLookUp)):
            temp[i][j] = float(ret[i][j][0]) / ret[i][j][1]
            if temp[i][j] == 0:
                temp[i][j] = np.nan
    return temp, user, movie

def prepareSVD():
    user = {}
    movie = {}
    ptr = 0
    with open("../files/movie.txt", "r") as f:
        for l in f :
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            movie[int(l[0])] = ptr
            ptr += 1
    ptr = 0
    with open("../files/user.txt", "r") as f:
        for l in f :
            if l[-1] == '\n' or l[-1] == '\r':
                l = l[:-1]
            l = l.split(",")
            user[int(l[0])] = ptr
            ptr += 1
    train = readTrain()
    ret = np.zeros((len(user), len(movie)))
    for item in train:
        ret[user[item[0]]][movie[item[1]]] = item[2]
    for i in range(len(user)):
        for j in range(len(movie)):
            if ret[i][j] == 0 :
                ret[i][j] = np.nan
    return ret, user, movie

def readMovie(numberOfIntervals = 5, minAge = 1919, maxAge = 2000):
    ret = {}
    interval = float(maxAge - minAge + 1) / numberOfIntervals
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
    print maxGen
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
    X = split_data(X, num_folds = 5)
    Y = split_data(Y, num_folds = 5)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def get_test(user, movie, test):
    X = []
    for item in test :
        app = [item[0], [user[item[1]] + movie[item[2]]]]
        X.append(app)
    return X


