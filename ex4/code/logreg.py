import numpy as np

def plot2D(ax, X, y, theta, name):
    '''
    Visualize decision boundary and data classes in 2D
    :param ax: matplotlib
    :param X: data
    :param y: data labels
    :param theta: model parameters
    :param name:
    :return:
    '''
    x1 = np.array(X[1,:])  # note: X_train[0,:] is the added row of 1s (bias)
    x2 = np.array(X[2,:])
    posterior1 = LOGREG().activationFunction(theta, X)
    posterior1 = np.squeeze(np.asarray(posterior1))
    markers = ['o','+']
    groundTruthLabels = np.unique(y)
    for li in range(len(groundTruthLabels)):
        x1_sub = x1[y[:] == groundTruthLabels[li]]
        x2_sub = x2[y[:] == groundTruthLabels[li]]
        m_sub = markers[li]
        posterior1_sub = posterior1[y[:] == groundTruthLabels[li]]
        ax.scatter(x1_sub, x2_sub, c = posterior1_sub, vmin = 0, vmax = 1 , marker = m_sub, label = 'ground truth label = ' + str(li))
    cbar = ax.colorbar()
    cbar.set_label('posterior value')
    ax.legend()
    x = np.arange(x1.min(),x1.max(),0.1)
    pms = [[0.1, 'k:'], [0.25, 'k--'], [0.5, 'r'], [0.75, 'k-.'], [0.9, 'k-']]
    for (p,m) in pms:
        yp = (- np.log((1/p)-1) - theta[1] * x - theta[0]) / theta[2]
        yp = np.squeeze(np.asarray(yp))
        ax.plot(x, yp, m, label = 'p = ' + str(p) )
        ax.legend()
    ax.xlabel('feature 1')
    ax.ylabel('feature 2')
    ax.title(name + '\n Posterior for class labeled 1')


def plot3D(ax, sub3d, X, y, theta , name):
    '''
    Visualize decision boundary and data classes in 3D
    :param ax:  matplotlib
    :param sub3d: fig.add_subplot(XXX, projection='3d')
    :param X: data
    :param y: data labels
    :param theta: model parameters
    :param name: plot name identifier
    :return:
    '''
    x1 = np.array(X[1, :])  # note: X_train[0,:] is the added row of 1s (bias)
    x2 = np.array(X[2, :])
    posterior1 = LOGREG().activationFunction(theta, X)
    posterior1 = np.squeeze(np.asarray(posterior1))
    markers = ['o', '+']
    groundTruthLabels = np.unique(y)
    for li in range(len(groundTruthLabels)):
        x1_sub = x1[y[:] == groundTruthLabels[li]]
        x2_sub = x2[y[:] == groundTruthLabels[li]]
        m_sub = markers[li]
        posterior1_sub = posterior1[y[:] == groundTruthLabels[li]]
        sub3d.scatter(x1_sub, x2_sub, posterior1_sub, c = posterior1_sub, vmin=0, vmax=1, marker=m_sub,label='ground truth label = ' + str(li))
    ax.legend()
    x = np.arange(x1.min(), x1.max(), 0.1)
    pms = [[0.1, 'k:'], [0.25, 'k--'], [0.5, 'r'], [0.75, 'k-.'], [0.9, 'k-']]
    for (p, m) in pms:
        yp = (- np.log((1 / p) - 1) - theta[1] * x - theta[0]) / theta[2]
        yp = np.squeeze(np.asarray(yp))
        z = np.ones(yp.shape) * p
        sub3d.plot(x, yp, z, m, label = 'p = ' + str(p))
        ax.legend()
    ax.xlabel('feature 1')
    ax.ylabel('feature 2')
    ax.title(name + '\n Posterior for class labeled 1')


class LOGREG(object):
    '''
    Logistic regression class based on lecture slides 2018-10-08
    '''
    def __init__(self, regularization = 0):
        self.r = regularization
        self._threshold = 1e-10


    def activationFunction(self, theta, X):
        # TODO: Implement logistic function
        # Maybe its 1/(1 + np.exp(-(theta.T * X) + theta))
        return 1/(1 + np.exp(-(theta.T * X)))


    def _costFunction(self, theta, X, y):
        '''
        Compute the cost function for the current model parameters
        :param theta: current model parameters
        :param X: data
        :param y: data labels
        :return: cost
        '''
        # TODO: Implement equation of cost function for posterior p(y=1|X,w)
        cost = 0
        for i in range(len(y)):
            cost += y[i]*np.log(self.activationFunction(theta, X[:,i])) + (1-y[i])*np.log(1-self.activationFunction(theta, X[:,i]))
        regularizationTerm = 0
        return cost + regularizationTerm


    def _calculateDerivative(self, theta, X, y):
        '''
        Compute the derivative of the model parameters
        :param theta: current model parameters
        :param X: data
        :param y: data labels
        :return: first derivative of the model parameters
        '''
        # TODO: Calculate derivative of loglikelihood function for posterior p(y=1|X,w)
        firstDerivative = 0
        for i in range(y.shape[1]):
            firstDerivative += (y[:,i]-self.activationFunction(theta, X[:,i])) * X[:,i].T
        regularizationTerm = 0
        return firstDerivative + regularizationTerm


    def _calculateHessian(self, theta,  X):
        '''
        :param theta: current model parameters
        :param X: data
        :return: the hessian matrix (second derivative of the model parameters)
        '''
        # TODO: Calculate Hessian matrix of loglikelihood function for posterior p(y=1|X,w)
        hessian = 0
        for i in range(X.shape[1]):
            hessian += X[:,i]*X[:,i].T * (self.activationFunction(theta,X[:,i])*(1-self.activationFunction(theta,X[:,i])))[0,0]
        regularizationTerm = 0
        return (- hessian + regularizationTerm)


    def _optimizeNewtonRaphson(self, X, y, niterations):
        '''
        Newton Raphson method to iteratively find the optimal model parameters (theta)
        :param X: data
        :param y: data labels (0 or 1)
        :param niterations: number of iterations to take
        :return: model parameters (theta)
        '''
        # TODO: Implement Iterative Reweighted Least Squares algorithm for optimization, use the calculateDerivative and calculateHessian functions you have already defined above
        theta = np.matrix(np.zeros((X.shape[0], 1))) # Initializing the theta vector as a numpy matrix class instance
        for n in range(niterations):
            hessianinverse = np.linalg.inv(self._calculateHessian(theta, X))
            hessianinverse[:,0] = 0
            hessianinverse[0,:] = 0
            deriv = self._calculateDerivative(theta, X, y)
            deriv[:,0] = 1
            theta = theta - hessianinverse * deriv.T
            loglikelihood = 0
            for i in range(len(y)):
                loglikelihood += int(y[:, i]) * theta.T * X[:, i] - np.log(1+np.exp(theta.T * X[:, i]))
            if loglikelihood < self._threshold:
                break
        return theta
        # note maximize likelihood (should become larger and closer to 1), maximize loglikelihood( should get less negative and closer to zero)


    def train(self, X, y, iterations):
        '''
        :param X: dataset
        :param y: ground truth labels
        :param iterations: Number of iterations to train
        :return: trained theta parameter
        '''
        X = np.matrix(X)
        y = np.matrix(y)
        self.theta = self._optimizeNewtonRaphson(X, y, iterations)
        return self.theta


    def classify(self, X):
        '''
        Classify data given the trained logistic regressor - access the theta parameter through self.
        :param x: Data to be classified
        :return: List of classification values (0.0 or 1.0)
        '''
        # TODO: Implement classification function for each entry in the data matrix
        predictions = self.activationFunction(self.theta,X)
        return predictions


    def printClassification(self, X, y):
        '''
        Calls classifyLinear and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement print classification
        N = X.shape[1]
        # TODO: change the values!
        predictions = self.classify(X)
        numOfMissclassified = 0
        for i in range(N):
            if (predictions[:,i] <= 0.5 and y[i] == 1) or (predictions[:,i] > 0.5 and y[i] == 0):
                numOfMissclassified +=1
        totalError = (numOfMissclassified/N)*100

        print("Total error: {:.2f}%, {}/{} misclassified".format(totalError, numOfMissclassified, N))
