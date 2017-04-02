import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from functools import partial
from preprocess import *
from sklearn.preprocessing import Imputer

#cited from http://stackoverflow.com/questions/35577553/how-to-fill-nan-values-in-numeric-array-to-apply-svd
def trainSVD(mat, iter = 50, compo = None, tolerance = 1e-4):
    if compo is None:
        svd = partial(np.linalg.svd, full_matrices = False)
    else :
        svd = partial(svds, k = compo)
    mu_hat = np.nanmean(mat, axis = 0, keepdims = 1)
    valid = np.isfinite(mat)
    Y_hat = np.where(valid, mat, mu_hat)
    iteration = 0
    v_prev = 0
    while iteration < iter:
        iteration += 1
        U, s, Vt = svd(Y_hat - mu_hat)
        Y_hat[~valid] = (U.dot(np.diag(s)).dot(Vt) + mu_hat)[~valid]

        mu_hat = Y_hat.mean(axis = 0, keepdims = 1)
        v = s.sum()
        if ((v - v_prev) / v_prev) < tolerance:
            break
    return Y_hat, mu_hat, U, s, Vt

def clip(x):
    if x < 1 : x = 1
    if x > 5 : x = 5
    return x

def main():
    imp = Imputer(strategy = 'mean', axis  =1)
    trainDirect, userDirect, movieDirect = prepareSVD()
#    trainFeature, userFeature, movieFeature =  prepareSpecialSVD()
    print trainDirect
#    direct, a,b,c,d = trainSVD(trainDirect)
    print trainDirect.shape
    direct = imp.fit_transform(trainDirect) 
    print direct.shape
#    feature, a,b,c,d= trainSVD(trainFeature)
    print direct
    test = readTest()
    out = open("../out/outSVD.txt", "w")
    out.write("Id,rating\n")
    for item in test :
        out.write(str(item[0]))
        out.write(", ")
        out.write(str(clip(int(direct[userDirect[item[1]]][movieDirect[item[2]]] + 0.5))) + "\n")
    out.close()
    return


main()
