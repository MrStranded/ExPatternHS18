import sys
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from svm import SVM, plot_data, plot_linear_separator



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
    image_side_length = np.sqrt(len(data[:,0])).astype(np.int64)
    fig = plt.figure(figsize=(8, 8))
    plt.title(name)
    correct = 0
    wrong = 0

    row_image_correct = []
    row_image_correct_temp = []
    row_image_wrong = []
    row_image_wrong_temp = []

    for i in range(dim):
        img = np.array(data[:, i]).reshape(image_side_length, image_side_length)
        if labels[0, i] == predictions[i]:
            #print(i, "correct")
            correct += 1
            if len(row_image_correct_temp ) == 0:
                row_image_correct_temp = img
            else:
                row_image_correct_temp = np.concatenate((row_image_correct_temp, img), axis=1)
                if np.mod(correct,image_side_length) == 0:
                    if len(row_image_correct) == 0:
                        row_image_correct = row_image_correct_temp
                    else:
                        row_image_correct = np.concatenate((row_image_correct, row_image_correct_temp),axis=0)
                    row_image_correct_temp = []
        else:
            #print(i, "wrong!")
            wrong += 1
            if len(row_image_wrong_temp) == 0:
                row_image_wrong_temp = img
            else:
                row_image_wrong_temp = np.concatenate((row_image_wrong_temp, img), axis=1)
                if np.mod(wrong,image_side_length) == 0:
                    if len(row_image_wrong) == 0:
                        row_image_wrong = row_image_wrong_temp
                    else:
                        row_image_wrong = np.concatenate((row_image_wrong, row_image_wrong_temp), axis=0)
                    row_image_wrong_temp = []

    empty = np.array(np.zeros((image_side_length, image_side_length)))
    n, k = row_image_correct_temp.shape
    k = int(np.subtract(image_side_length, np.divide(k,16)))
    # Fill the image row up with empty images
    if k != 0:
        for i in range(k):
            row_image_correct_temp = np.concatenate((row_image_correct_temp, empty), axis=1)
        row_image_correct = np.concatenate((row_image_correct, row_image_correct_temp), axis=0)

    n, k = row_image_wrong_temp.shape
    k = int(np.subtract(image_side_length, np.divide(k, 16)))
    # Fill the image row up with empty images
    if k != 0:
        for i in range(k):
            row_image_wrong_temp = np.concatenate((row_image_wrong_temp, empty), axis=1)
        if len(row_image_wrong) == 0:
            row_image_wrong = row_image_wrong_temp
        else:
            row_image_wrong = np.concatenate((row_image_wrong, row_image_correct_temp), axis=0)

    fig.add_subplot(1, 2, 1).set_title("Correct Classification")
    plt.imshow(row_image_correct)
    if len(row_image_wrong) != 0:
        fig.add_subplot(1, 2, 2).set_title("Wrong Classification")
        plt.imshow(row_image_wrong)
    plt.axis('off')
    plt.show()


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
    C = 1000
    svm = SVM(C)
    svm.train(train_x, train_label, 'poly') # 'rbf' is taking forever to compute

    '''
    Results for rbf with different sigma values:
    sigma = 0.1:
        13:
            Number of support vectors: 1663
            Train error: 0.00%
            Test error: 38.60%
        38:
            Number of support vectors: 1200
            Train error: 0.00%
            Test error: 50.00%
    sigma = 1:
        13:
            Number of support vectors: 870
            Train error: 0.00%
            Test error: 6.05%
        38:
            Number of support vectors: 1200
            Train error: 0.00%
            Test error: 48.80%
    sigma = 10:
        13:
            Number of support vectors: 32
            Train error: 0.00%
            Test error: 0.70%
        38:
            Number of support vectors: 114
            Train error: 0.00%
            Test error: 2.71%
    -------------------------------- different values for C
    sigma = 10 and C = 1:
        13:
            Number of support vectors: 69
            Train error: 0.06%
            Test error: 0.70%
        38:
            Number of support vectors: 267
            Train error: 1.00%
            Test error: 3.92%
    sigma = 10 and C = 10:
        13:
            Number of support vectors: 33
            Train error: 0.00%
            Test error: 0.70%
        38:
            Number of support vectors: 135
            Train error: 0.08%
            Test error: 2.71%
    sigma = 10 and C = 100: (C = 1000 gives same result)
        13:
            Number of support vectors: 31
            Train error: 0.00%
            Test error: 0.70%
        38:
            Number of support vectors: 117
            Train error: 0.00%
            Test error: 2.71%
    '''

    # training predictions
    predictions_train = svm.classifyKernel(train_x)
    # testing predictions
    predictions_test = svm.classifyKernel(test_x)

    print("Training error")
    # TODO: Compute training error of SVM
    #svm.printKernelClassificationError(train_x,train_label) # we don't do that here because then we'd do the classification twice
    #svm.printLinearClassificationError(train_x,train_label)
    dim = train_label.shape[1]
    wrong = 0
    for i in range(dim):
        if train_label[0,i] * predictions_train[i] < 0:
            wrong += 1
    print("Train error: {:.2f}%".format(wrong/dim*100))

    print("Test error")
    # TODO: Compute test error of SVM
    #svm.printKernelClassificationError(test_x,test_label)
    #svm.printLinearClassificationError(test_x,test_label)
    dim = test_label.shape[1]
    wrong = 0
    for i in range(dim):
        if test_label[0,i] * predictions_test[i] < 0:
            wrong += 1
    print("Test error: {:.2f}%".format(wrong/dim*100))

    # TODO: Visualize classification - correct and wrongly classified images
    visualizeClassification(test_x, test_label, predictions_test, 2, "Test classification")

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