#This file aims to provide an interface for solving the orthogonal procrustes problem.
#Given two sets of points, it finds the optimal orthogonal transformation that best aligns the two sets in a least-squares sense.

#Notably, this will also be extensible heauristically to euclidean (In the sense of the Euclidean group on R^d) procrustes, which allows for translation
#Though the heuristic method works perfectly in the realisable case, it is a solid heuristic in the non-realisable case as well.
import numpy as np

def normalise_expectations(X,Y):
    #X is a matrix of shape (n_samples, n_features)
    X_mean = X.mean(axis=1)
    Y_mean = Y.mean(axis=1)
    Y_out = Y - Y_mean[:, np.newaxis]
    X_out = X - X_mean[:, np.newaxis]
    return X_out, Y_out, X_mean, Y_mean

def procrustes(X,Y):
    #Assumes X and Y are already normalised to have mean 0 
    assert X.shape == Y.shape, "X and Y must have the same shape"
    assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray), "X and Y must be numpy arrays"
    M = Y @ X.T

    #We must find R such that ||R - M||_F is minimised, where R is orthogonal
    U,Sigma,Vt = np.linalg.svd(M)
    R = U @ Vt
    return R

def euclidean_procrustes(X,Y):
    #This is a heuristic method for solving the euclidean procrustes problem, which allows for translation as well as rotation
    X_out, Y_out, X_mean, Y_mean = normalise_expectations(X,Y)
    R = procrustes(X_out, Y_out)
    #Now we must find the translation vector t such that ||R @ X + t - Y||_F is minimised
    t = Y_mean - R @ X_mean
    return R, t

def impose_X_on_Y(X,Y):
    R,t = euclidean_procrustes(X,Y)
    Y_aligned = R @ X + t[:, np.newaxis]
    return Y_aligned
    
#sanity_check
if __name__ == "__main__":
    X = np.random.rand(2304, 5_000)
    #R_true is just a permutation matrix
    permutation = np.random.permutation(2304)
    R_true = np.zeros((2304, 2304))
    for i in range(2304):
        R_true[i, permutation[i]] = 1
    t_true = np.random.rand(2304)
    Y = R_true @ X + t_true[:, np.newaxis]
    Y_aligned = impose_X_on_Y(X,Y)

    #check they are close
    assert np.allclose(Y_aligned, Y, atol=1e-5), "Y_aligned and Y are not close enough"
    print("Sanity check passed!")

