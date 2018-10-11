import numpy as np
from scipy.stats import multivariate_normal


class MVND:
    def __init__(self, data, p=1.0):
        self.p = p
        a = np.array([[1, 2], [3, 4]])
        self.data = data
        self.mean = data.mean(1)
        self.cov = np.cov(data)

    def pdf(self, x):
        return multivariate_normal.pdf(x, self.mean, self.cov)
    
    def logpdf(self, x):
        return multivariate_normal.logpdf(x, self.mean, self.cov)
