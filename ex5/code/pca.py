import numpy as np
import matplotlib.pyplot as plt
import math
import json

class PCA():
    '''
    Principal Component Analysis
    Specify maximum number of components in the construction (__init__)
    '''
    def __init__(self, maxComponents = -1):
        self._maxComponents = maxComponents


    def plot_pca(self, X, maxxplot=200):
        """
        Plot pca data and first 2 principal component directions
        Used to visualize the toy dataset
        """
        vec1len = math.sqrt(self.S[0])
        vec2len = math.sqrt(self.S[1])
        # Take random subset from X for plotting (max 200)
        scat = X[:, np.random.permutation(np.min((maxxplot, X.shape[1])))]
        plt.scatter(scat[0, :], scat[1, :])
        plt.quiver(self.mu[0], self.mu[1], self.U[0, 0] * vec1len, self.U[0, 1] * vec1len, angles='xy', scale_units='xy', scale=1)
        plt.quiver(self.mu[0], self.mu[1], self.U[1, 0] * vec2len, self.U[1, 1] * vec2len, angles='xy', scale_units='xy', scale=1)
        plt.grid()


    def train(self, X):
        '''
        Compute PCA "manually" by using SVD
        Refer to the LinearTransform slides for details
        NOTE: Remember to set svd(, full_matrices=False)!!!
        :param X: Training data
        '''
        # TODO: Implement PCA mean (mu), principal components (U) and variance (S) of the given data (X)
        mu = np.mean(X,axis=1)
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        if self._maxComponents == -1:
            # Use all principal components
            m = s.shape[0]
        else:
            # only use self._maxComponents
            m = self._maxComponents
            s[m:] = 0
            u[:, m:] = 0
        # nxm matrix which stores m principal components
        # is a vector where the i-th entry contains the i-th variance value lambda corresponding to the i-th
        # principal component
        self.mu = mu
        self.U  = u
        self.S  = s
        return (mu, self.U, self.S)




    def to_pca(self, X):
        '''
        :param X: Data to be projected into PCA space
        :return: alpha - feature vector
        '''
        #TODO: Exercise 1
        # Move center of mass to origin
        # Build data matrix, from mean free data
        dataOrigin = X
        for i in range(X.shape[1]):
            dataOrigin[:, i] -= self.mu
        alpha = np.dot(dataOrigin.T,self.U)
        return alpha


    def from_pca(self, alpha):
        '''
        :param alpha: feature vector
        :return: X in the original space
        '''
        # TODO: Exercise 1
        Xout =  np.dot(alpha.T,self.U.T)
        for i in range(alpha.shape[1]):
            Xout[:,i] += self.mu
        return Xout


    def project(self, X, k):
        '''
        :param X: Data to be projected into PCA space
        :param k: Dimensionality the projection should be limited to
        :return: projected data (x
        '''
        # TODO: Exercise 1
        self._maxComponents = k
        self.train(X)
        x_projected = self.to_pca(X)
        return x_projected