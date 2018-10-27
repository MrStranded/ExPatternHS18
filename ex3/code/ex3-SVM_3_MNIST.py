import sys
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from svm import SVM



def visualizeClassification(data, labels, predictions, num, name=''):
    '''
    Use SVM classifier to classify images and plot a window with correctly classified and one with wrongly classified images
    :param data: MNIST data each column is an image
    :param labels: Data labels (-1.0 or 1.0)
    :param predictions: Predicted data labels (-1.0 or 1.0)
    :param num: Number of MNIST images to show
    :param name: Optional name of the plot
    '''
    # TODO: Implement visualization function
    dim = data.shape[1]
    for i in range(dim):
        if labels[0,i] == predictions[i,0]:
            print(i, "correct")
        else:
            print(i, "wrong!")


def svmMNIST(train, test):
    '''
    Train an SVM with the given training data and print training + test error
    :param train: Training data
    :param test: Test data
    :return: Trained SVM object
    '''

    train_label = np.transpose(train[0,:].astype(np.double)[:,None])
    train_x     = train[1:,:].astype(np.double)

    test_label = np.transpose(test[0,:].astype(np.double)[:,None])
    test_x     = test[1:,:].astype(np.double)

    # TODO: Train svm
    C = None
    svm = SVM(C)
    svm.train(train_x, train_label, 'linear', 2)

    print("Training error")
    # TODO: Compute training error of SVM
    svm.printKernelClassificationError(train_x,train_label)
    #svm.printLinearClassificationError(train_x,train_label)

    print("Test error")
    # TODO: Compute test error of SVM
    svm.printKernelClassificationError(test_x,test_label)
    #svm.printLinearClassificationError(test_x,test_label)

    # TODO: Visualize classification - correct and wrongly classified images
    # training predictions
    predictions_train = svm.classifyKernel(train_x)
    visualizeClassification(train_x, train_label, predictions_train, 2)

    # testing predictions
    predictions_test = svm.classifyKernel(test_x)
    visualizeClassification(test_x, test_label, predictions_test, 2)

    return svm


def testMNIST13():
    '''
     - Load MNIST dataset, characters 1 and 3
     - Train a kernel SVM
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running MNIST13")
    toy = scipy.io.loadmat('../data/zip13.mat')
    train = toy['zip13_train']
    test = toy['zip13_test']
    svmMNIST(train, test)

def testMNIST38():
    '''
     - Load MNIST dataset, characters 3 and 8
     - Train a kernel SVM
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running MNIST38")
    toy = scipy.io.loadmat('../data/zip38.mat')
    train = toy['zip38_train']
    test = toy['zip38_test']
    svmMNIST(train, test)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nSVM exercise - MNIST Example")
    print("##########-##########-##########")
    testMNIST13()
    print("##########-##########-##########")
    testMNIST38()
    print("##########-##########-##########")