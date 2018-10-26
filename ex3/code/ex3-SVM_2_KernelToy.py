import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from svm import SVM, plot_data, plot_kernel_separator


def svmKernelToyExample():
    '''
     - Load non-linear separable toy dataset
     - Train a kernel SVM
     - Print training and test error
     - Plot data and separator
    '''
    toy_train_x = np.load('../data/inputs.npy') #toy['toy_train']
    toy_train_label = np.load('../data/targets.npy').astype(float)
    toy_train_label[toy_train_label==0.0] = -1.0

    toy_test_x = np.load('../data/validation_inputs.npy') #toy['toy_train']
    toy_test_label = np.load('../data/validation_targets.npy').astype(float)
    toy_test_label[toy_test_label==0.0] = -1.0

    # TODO: Train svm
    svm = SVM(10000)
    svm.train(toy_train_x,toy_train_label,'rbf',2)

    print("Training error")
    # TODO: Compute training error of SVM
    svm.printKernelClassificationError(toy_train_x,toy_train_label)
    print("Test error")
    # TODO: Compute test error of SVM
    svm.printKernelClassificationError(toy_test_x,toy_test_label)
    print("Visualizing data")
    # TODO: Visualize data and separation boundary - hint: you can use the given "plot_kernel_separator" and the "plot_data" functions
    plot_data(plt, toy_test_x, toy_test_label, [['red', '+'], ['blue', '_']])
    plot_kernel_separator(plt, svm, 0, 1)
    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nSVM exercise - Non-linear Toy Example")
    print("##########-##########-##########")
    svmKernelToyExample()
    print("##########-##########-##########")