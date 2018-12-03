import numpy as np
import matplotlib.pyplot as plt
import math
import json


class PCA():
    '''
    Principal Component Analysis
    Specify maximum number of components in the construction (__init__)
    '''

    def __init__(self, maxComponents=-1):
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
        plt.quiver(self.mu[0], self.mu[1], self.U[0, 0] * vec1len, self.U[0, 1] * vec1len, angles='xy',
                   scale_units='xy', scale=1)
        plt.quiver(self.mu[0], self.mu[1], self.U[1, 0] * vec2len, self.U[1, 1] * vec2len, angles='xy',
                   scale_units='xy', scale=1)
        plt.grid()
        plt.axis('equal')

    def train(self, X):
        '''
        Compute PCA "manually" by using SVD
        Refer to the LinearTransform slides for details
        NOTE: Remember to set svd(, full_matrices=False)!!!
        :param X: Training data
        '''
        # TODO: Implement PCA mean (mu), principal components (U) and variance (S) of the given data (X)
        [rows, cols] = X.shape
        mu = (np.mean(X, axis=1)).reshape((rows, 1))
        mu_expanded = np.outer(mu, np.ones(X.shape[1]))
        X = X - mu_expanded
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        if self._maxComponents == -1:
            # Use all principal components
            m = s.shape[0]
        else:
            # only use self._maxComponents
            m = self._maxComponents
            s = s[:m]
            u = u[:, :m]
        # nxm matrix which stores m principal components
        # is a vector where the i-th entry contains the i-th variance value lambda corresponding to the i-th
        # principal component
        self.mu = mu
        self.U = u
        self.S = s
        return mu, self.U, self.S

    def to_pca(self, X):
        '''
        :param X: Data to be projected into PCA space
        :return: alpha - feature vector
        '''
        # TODO: Exercise 1
        # Move center of mass to origin
        # Build data matrix, from mean free data
        mu_expanded = np.outer(self.mu, np.ones(X.shape[1]))
        X = X - mu_expanded
        alpha = np.dot(self.U.T, X)
        return alpha

    def from_pca(self, alpha):
        '''
        :param alpha: feature vector
        :return: X in the original space
        '''
        # TODO: Exercise 1
        alpha_reshaped = alpha.reshape((self.U.shape[1], -1))
        x_out = self.U.dot(alpha_reshaped) + self.mu
        return x_out

    def project(self, X, k):
        '''
        :param X: Data to be projected into PCA space
        :param k: Dimensionality the projection should be limited to
        :return: projected data (x
        '''
        # TODO: Exercise 1
        self._maxComponents = k
        x_projected = self.to_pca(X)
        x_projected[k:, :] = 0
        x_projected = self.from_pca(x_projected)
        return x_projected
