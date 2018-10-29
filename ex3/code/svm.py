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
        return np.dot(x1,x2)

    def __polynomialKernel__(self, x1, x2, p):
        # TODO: Implement polynomial kernel function
        return np.power(np.dot(x1.T,x2)+1, p)

    def __gaussianKernel__(self, x1, x2, sigma):
        # TODO: Implement gaussian kernel function
        # This vile distinction is necessary to serve the different ways in which __computeKernerl__() is called
        # Normally x2 has the shape (256), but in the calculation for the bias it is (256,1663) ... :P
        if x2.shape.__len__() == 1:
            return np.exp(-np.power(norm(x1 - x2), 2) / (2 * np.power(sigma, 2)))
        else:
            dx = np.tile(x1.reshape((x1.shape[0], 1)), (1, x2.shape[1])) - x2
            return np.exp(-(np.power(norm(dx, axis=0), 2) / (2 * np.power(sigma, 2))))

        #dx = x1 - x2 #np.tile(x1.reshape((x1.shape[0], 1)), (1, x1.shape[0])) - x2
        #return np.exp(-(np.power(norm(dx, axis=0), 2) / (2 * np.power(sigma, 2))))
        #return np.exp(-np.power(norm(x1-x2),2) / (2*np.power(sigma,2)))

    def __computeKernel__(self, x, kernelFunction, pars):
        # TODO: Implement function to compute the kernel matrix
        dim = x.shape[1]
        K = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                K[i,j] = kernelFunction(x[:,i],x[:,j],pars)

        return K


    def train(self, x, y, kernel=None, kernelpar=2):
        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar

        NUM = x.shape[1]

        P = np.zeros((NUM, NUM))
        for i in range(NUM):
            for j in range(NUM):
                P[i, j] = y[0,i]*y[0,j] * np.dot(x[:,i], x[:,j])

        # we'll solve the dual
        # obtain the kernel
        if kernel == 'linear':
            print('Fitting SVM with linear kernel')
            self.kernel = self.__linearKernel__
            K = self.__computeKernel__(x,self.kernel,kernelpar)
        elif kernel == 'poly':
            print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
            self.kernel = self.__polynomialKernel__
            K = self.__computeKernel__(x,self.kernel,kernelpar)
        elif kernel == 'rbf':
            print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
            self.kernel = self.__gaussianKernel__
            K = self.__computeKernel__(x,self.kernel,kernelpar)
        else:
            print('Fitting linear SVM')
            K = np.zeros(shape=(NUM, NUM))

            for i in range(NUM):
                for j in range(NUM):
                    K[i, j] = np.dot(x[:, i], x[:, j])

        if self.C is None:
            G = cvx.matrix(-np.eye(NUM))
            h = cvx.matrix(np.zeros(NUM))
        else:
            print("Using Slack variables")
            G = cvx.matrix(np.concatenate((-np.eye(NUM), np.eye(NUM))))
            h = cvx.matrix(np.concatenate((np.zeros(NUM), np.ones(NUM)*self.C)))

        P = cvx.matrix(np.zeros((NUM, NUM)))

        for i in range(NUM):
            for j in range(NUM):
                P[i, j] = K[i, j] * np.dot(y[:, i], y[:, j])

        q = cvx.matrix(np.ones(NUM) * (-1))
        A = cvx.matrix(y)
        b = cvx.matrix(0.0)

        cvx.solvers.options["show_progress"] = False
        solution = cvx.solvers.qp(P, q, G, h, A, b)
        sol_x = np.array(solution["x"]).transpose()
        self.lambda_indices = np.flatnonzero(sol_x > self.__TOL)
        self.lambdas = sol_x[:, self.lambda_indices]  # Only save > 0
        self.sv = x[:, self.lambda_indices]  # List of support vectors
        self.sv_labels = y[:, self.lambda_indices]  # List of labels for the support vectors (-1 or 1 for each support vector)
        print("Number of support vectors:",self.sv.shape[1])

        if (kernel is None):
            #self.bias = np.mean(self.sv_labels - self.w.dot(self.sv))  # Bias
            self.w = np.sum(self.lambdas * self.sv_labels * self.sv, axis=1)  # SVM weights
            self.bias = np.mean(self.sv_labels - self.w.dot(self.sv))  # Bias
        else:
            wx = np.zeros(self.sv.shape[1])
            for i in range(self.lambdas.shape[1]):
                wx += self.lambdas[0, i] * self.sv_labels[0, i] * self.kernel(self.sv[:, i], self.sv, self.kernelpar)
            self.bias = np.mean(self.sv_labels - wx)  # Bias

    def classifyLinear(self, x):
        '''
        Classify data given the trained linear SVM - access the SVM parameters through self.
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # TODO: Implement
        result = []
        for i in range(x.shape[1]):
            result.append(np.dot(np.array([self.w[:]]),x[:,i]) + self.bias)
        return result

    def printLinearClassificationError(self, x, y):
        '''
        Calls classifyLinear and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        classification = self.classifyLinear(x)
        equals = 0
        for i in range(classification.__len__()):
            if (classification[i] > 0 and y[0, i] > 0):
                equals += 1
            elif (classification[i] < 0 and y[0, i] < 0):
                equals += 1
            else:
                continue
        result = ((classification.__len__() - equals) / classification.__len__()) * 100
        print("Total error: {:.2f}%".format(result))

    def classifyKernel(self, x):
        '''
        Classify data given the trained kernel SVM - use self.kernel and self.kernelpar to access the kernel function and parameter
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # TODO: Implement
        t = np.zeros(x.shape[1])
        for i in range(self.lambdas.shape[1]):
            t += self.lambdas[0, i] * self.sv_labels[0, i] * self.kernel(self.sv[:, i], x, self.kernelpar)
        return np.sign(t + self.bias)

    def printKernelClassificationError(self, x, y):
        '''
        Calls classifyKernel and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        classification = self.classifyKernel(x)
        equals = 0
        for i in range(classification.__len__()):
            if (classification[i] * y[0, i] > 0):
                equals += 1
        result = ((classification.__len__() - equals) / classification.__len__()) * 100
        print("Total error: {:.2f}%".format(result))