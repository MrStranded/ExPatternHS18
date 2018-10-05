import numpy as np
from scipy.stats import multivariate_normal


class MVND:
    def __init__(self, data, p=1.0):
        self.p = p
        self.data = data
        self.mean = np.matrix(data).mean(1)
        self.cov  = np.cov(data)
        self.prefactor = 1/(np.sqrt(np.power(2*np.pi, np.shape(self.data)[1])*np.linalg.det(self.cov)))

    def pdf(self, x):
        exponent = np.power(np.e, -0.5*(x-self.data).T.dot(np.linalg.inv(self.cov)).dot(x-self.data))
        return self.prefactor*exponent
    
    def logpdf(self, x):
        return np.log(self.prefactor) + (-0.5*(x-self.data).T.dot(np.linalg.inv(self.cov)).dot(x-self.data))
