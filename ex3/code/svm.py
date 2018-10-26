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
        return np.power(np.dot(x1,x2)+1, p)

    def __gaussianKernel__(self, x1, x2, sigma):
        # TODO: Implement gaussian kernel function
        return np.exp(-np.power(norm(x1-x2),2)/(2*np.power(sigma,2)))


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
                val = y[0,i]*y[0,j] * np.dot(x[:,i], x[:,j])
                P[i,j] = val
        print(P)

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
        self.lambdas = np.flatnonzero(sol_x > self.__TOL)  # Only save > 0
        self.sol_x = sol_x[:,self.lambdas]
        self.sv = x[:, self.lambdas]  # List of support vectors
        self.sv_labels = y[:, self.lambdas]  # List of labels for the support vectors (-1 or 1 for each support vector)
        self.w = np.sum(sol_x[:, self.lambdas] * self.sv_labels * self.sv, axis=1)  # SVM weights

        if (kernel == None):
            self.bias = np.mean(self.sv_labels - self.w.dot(self.sv))  # Bias
        else:
            wx = np.zeros(self.sv.shape[1])
            for i in range(self.lambdas.__len__()):
                kernelsum = 0

                for j in range(self.sv.shape[0]):
                    kernelsum += self.kernel(self.sv[:,i],self.sv[:,j],self.kernelpar)

                wx += self.sol_x[0, i] * self.sv_labels[0, i] * kernelsum
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
            result.append(np.dot(np.array([self.w[:]]),x[:,i])+ self.bias)
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
        result = []
        for x_index in range(x.shape[1]):
            sum = 0
            for i in range(self.lambdas.__len__()):
                sum += self.sol_x[:,i]*self.sv_labels[:,i]*self.kernel(self.sv[:,i],x[:,x_index],self.kernelpar)
            result.append(sum + self.bias)
        return np.sign(np.array(result))

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
            if (classification[i] > 0 and y[0, i] > 0):
                equals += 1
            elif (classification[i] < 0 and y[0, i] < 0):
                equals += 1
            else:
                continue
        result = ((classification.__len__() - equals) / classification.__len__()) * 100
        print("Total error: {:.2f}%".format(result))