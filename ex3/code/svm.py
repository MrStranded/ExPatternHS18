import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import cvxopt as cvx


def plot_data(ax, x, y, STYLE, label=''):
    '''
    Visualize 2D data items - color according to their class
    :param ax: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param x: 2D data
    :param y: Data labels
    :param STYLE: Marker style and color in list format, ex: [['red', '+'], ['blue', '_']]
    :param label: Obtional plot name
    '''
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[:,y[0,:] == unique[li]]
        ax.scatter(x_sub[0,:], x_sub[1,:], c=STYLE[li][0], marker=STYLE[li][1], label=label+str(li))
    ax.legend()


def plot_linear_separator(ax, svm, datamin, datamax):
    '''
    Visualize linear SVM separator with margins
    :param ax: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param svm: SVM object
    :param datamin: min value on x and y axis to be shown
    :param datamax: max value on x and y axis to be shown
    '''
    x = np.arange(datamin, datamax+1.0)
    MARG = -(svm.w[0] * x + svm.bias) / svm.w[1]
    YUP = (1 - svm.w[0] * x - svm.bias) / svm.w[1]     # Margin
    YLOW = (-1 - svm.w[0] * x - svm.bias) / svm.w[1]   # Margin
    ax.plot(x, MARG, 'k-')
    ax.plot(x, YUP, 'k--')
    ax.plot(x, YLOW, 'k--')
    for sv in svm.sv:
        ax.plot(sv[0], sv[1], 'kx')


def plot_kernel_separator(ax, svm, datamin, datamax, h=0.05, alpha=0.25):
    '''
    :param ax: Plotting library to be used - ex pass plt (import matplotlib.pyplot as plt)
    :param svm: SVM object
    :param datamin: min value on x and y axis to be shown
    :param datamax: max value on x and y axis to be shown
    :param h: Density of classified background points
    :return:
    '''
    # function visualizes decision boundaries using color plots
    # creating meshgrid for different values of features
    xx, yy = np.meshgrid(np.arange(datamin, datamax, h), np.arange(datamin, datamax, h))
    # extracting predictions at different points in the mesh
    some = np.transpose(np.c_[xx.ravel(), yy.ravel()])
    Z = svm.classifyKernel(some)
    Z = Z.reshape(xx.shape)
    # plotting the mesh
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=alpha)
    for sv in svm.sv:
        ax.plot(sv[0], sv[1], 'kx')
    ax.grid()


class SVM(object):
    '''
    SVM class
    '''
    def __init__(self, C = None):
        self.C = C
        self.__TOL = 1e-5

    def __linearKernel__(self, x1, x2, _):
        # TODO: Implement linear kernel function
        return None

    def __polynomialKernel__(self, x1, x2, p):
        # TODO: Implement polynomial kernel function
        return None

    def __gaussianKernel__(self, x1, x2, sigma):
        # TODO: Implement gaussian kernel function
        return None


    def __computeKernel__(self, x, kernelFunction, pars):
        # TODO: Implement function to compute the kernel matrix
        return K


    def train(self, x, y, kernel=None, kernelpar=2):
        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar

        NUM = x.shape[1]

        # we'll solve the dual
        # obtain the kernel
        if kernel == 'linear':
            print('Fitting SVM with linear kernel')
            K = None
            self.kernel = self.__linearKernel__
        elif kernel == 'poly':
            print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
            K = None
            self.kernel = self.__polynomialKernel__
        elif kernel == 'rbf':
            print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
            K = None
            self.kernel = self.__gaussianKernel__
        else:
            print('Fitting linear SVM')
            K = None

        if self.C is None:
            G = None
            h = None
        else:
            print("Using Slack variables")
            G = None
            h = None


        # TODO: Compute below values according to the lecture slides
        self.lambdas = None # Only save > 0
        self.w = None # SVM weights
        self.sv = None # List of support vectors
        self.sv_labels = None # List of labels for the support vectors (-1 or 1 for each support vector)
        self.bias = None # Bias


    def classifyLinear(self, x):
        '''
        Classify data given the trained linear SVM - access the SVM parameters through self.
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # TODO: Implement
        return None

    def printLinearClassificationError(self, x, y):
        '''
        Calls classifyLinear and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        print("Total error: {:.2f}%".format(result))

    def classifyKernel(self, x):
        '''
        Classify data given the trained kernel SVM - use self.kernel and self.kernelpar to access the kernel function and parameter
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # TODO: Implement
        return None

    def printKernelClassificationError(self, x, y):
        '''
        Calls classifyKernel and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        print("Total error: {:.2f}%".format(result))
