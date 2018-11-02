import sys
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from logreg import LOGREG, plot2D, plot3D


def logregToy():
    #load data
    toy = scipy.io.loadmat('../data/toy.mat')
    toy_train = toy['toy_train']
    toy_test = toy['toy_test']

    toy_train_label = np.transpose(toy_train[0, :].astype(np.double))
    toy_train_label[toy_train_label < 0] = 0.0
    toy_train_x = toy_train[0:3, :].astype(np.double)
    toy_train_x[0,:] = 1.0 # adding row of 1s in X matrix to account for w0 term

    toy_test_label = np.transpose(toy_test[0, :].astype(np.double))
    toy_test_label[toy_test_label < 0] = 0.0
    toy_test_x = toy_test[0:3, :].astype(np.double)
    toy_test_x[0,:] = 1.0 # adding row of 1s in X matrix to account for w0 term

    #training coefficients
    regcoeff = [0, 0.1, 0.5]
    # without regularization : regcoeff = 0
    # with regularization    : regcoeff = 1 / 2*sigma^2

    for r in regcoeff:

        print('with regularization coefficient ' , r)
        logreg = LOGREG(r)
        trained_theta = logreg.train(toy_train_x, toy_train_label, 50)
        print("Training:")
        logreg.printClassification(toy_train_x, toy_train_label)
        print("Test:")
        logreg.printClassification(toy_test_x, toy_test_label)


        # plot for toy dataset
        figname =  'Toy dataset with r: {}'.format(r)
        fig = plt.figure(figname)
        plt.subplot(221)
        plot2D(plt, toy_train_x, toy_train_label, trained_theta, 'Training')
        plt.subplot(222)
        plot2D(plt, toy_test_x, toy_test_label, trained_theta, 'Testing')

        plot3D(plt, fig.add_subplot(223, projection='3d'), toy_train_x, toy_train_label, trained_theta, 'Training')
        plot3D(plt, fig.add_subplot(224, projection='3d'), toy_test_x, toy_test_label, trained_theta, 'Test')

    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nLOGREG exercise - Toy Example")
    print("##########-##########-##########")
    logregToy()
    print("##########-##########-##########")
