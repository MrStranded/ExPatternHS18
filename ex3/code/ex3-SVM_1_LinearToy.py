import sys
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from svm import SVM, plot_data, plot_linear_separator


def svmLinearToyExample():
    '''
     - Load linear separable toy dataset
     - Train a linear SVM
     - Print training and test error
     - Plot data and separator
    '''
    C = None

    toy = scipy.io.loadmat('../data/toy.mat')
    toy_train = toy['toy_train']
    toy_test = toy['toy_test']

    toy_train_label = np.transpose(toy_train[0,:].astype(np.double)[:,None])
    toy_train_x     = toy_train[1:3,:].astype(np.double)

    toy_test_label = np.transpose(toy_test[0,:].astype(np.double)[:,None])
    toy_test_x     = toy_test[1:3,:].astype(np.double)

    svm = SVM(C)
    svm.train(toy_train_x, toy_train_label)

    print("Training error")
    # TODO: Compute training error of SVM - hint: use the printLinearClassificationError from the SVM class
    svm.printLinearClassificationError(toy_train_x,toy_train_label)
    print("Test error")
    # TODO: Compute test error of SVM - hint: use the printLinearClassificationError from the SVM class
    svm.printLinearClassificationError(toy_test_x,toy_test_label)
    print("Visualizing data")
    # TODO: Visualize data and separation function - hint: you can use the given "plot_linear_separator" and the "plot_data" functions
    plot_linear_separator(plt,svm,-100,100)

if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nSVM exercise - Toy Example")
    print("##########-##########-##########")
    svmLinearToyExample()
    print("##########-##########-##########")
