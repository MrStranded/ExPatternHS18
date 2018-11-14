import sys
import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from logreg import LOGREG


def figurePlotting(imgarray, N, name='', random=True):
    '''
    MNIST image visualization - rescaling the vector images to 16x16 and visualizes in a matplotlib plot
    :param imgarray: Array of images to be visualized, each column is an image
    :param N: Number of images per row/column
    :param name: Optional name of the plot
    :param random: True if the images should be taken randomly from the array - otherwise start of the array is taken
    '''
    plt.figure(name)
    for i in range(0, N * N):
        imgIndex = i
        if random:
            high = imgarray.shape[1]
            if high == 0:
                continue
            imgIndex = np.random.randint(low=0, high=high)
        img = np.reshape(imgarray[:, imgIndex], (16, 16))
        plt.subplot(N, N, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')


def visualizeClassification(data, labels, predictions, num, name=''):
    '''
    Use LOGREG classifier to classify images and plot a window with correctly classified and one with wrongly classified images
    :param data: MNIST data each column is an image
    :param labels: Data labels (-1.0 or 1.0)
    :param predictions: Predicted data labels (-1.0 or 1.0)
    :param num: Number of MNIST images to show
    :param name: Optional name of the plot
    '''
    res = np.abs(predictions - labels)
    nummiss = int(np.sum(res))
    numcorr = int(data.shape[1]-nummiss)
    index = (res == 1.0).reshape(-1).astype(bool)

    xmiss = data[:,index]
    rowcol = int(math.ceil(math.sqrt(min(num, nummiss))))

    if rowcol > 0:
        figurePlotting(xmiss, rowcol, name+": Misclassified")

    index = np.invert(index)
    xcorr = data[:,index]
    rowcol = int(math.ceil(math.sqrt(min(num, numcorr))))

    if rowcol > 0:
        figurePlotting(xcorr, rowcol, name+": Correct")
    plt.show()


def logregMNIST(train, test):
    '''
    without reg : 0
    with reg: regcoeff = 1 / 2sigma^2
    :param train:
    :param test:
    :return:
    '''
    regcoeff = [0, 0.1, 0.5]

    train_label  = np.transpose(train[0,:].astype(np.double))
    train_label[train_label < 0] = 0.0
    train_x      = train[0:,:].astype(np.double)
    train_x[0,:] = 1.0

    test_label   = np.transpose(test[0,:].astype(np.double))
    test_label[test_label < 0] = 0.0
    test_x       = test[0:,:].astype(np.double)
    test_x[0, :] = 1.0

    for r in regcoeff:
        logreg = LOGREG(r)

        print('Training a LOGREG classifier with regularization coefficient: {}'.format(r))

        # training
        logreg.train(train_x, train_label, 50)
        print('Training')
        logreg.printClassification(train_x, train_label)
        print('Test')
        logreg.printClassification(test_x, test_label)

        visualizeClassification(train_x[1:,:], train_label, logreg.classify(train_x), 3 * 3, 'training with reg: {}'.format(r))
        visualizeClassification(test_x[1:,:], test_label, logreg.classify(test_x), 3 * 3, 'test with reg: {}'.format(r))


def testMNIST13():
    '''
     - Load MNIST dataset, characters 1 and 3
     - Train a logistic regression classifyer
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running MNIST13")
    toy = scipy.io.loadmat('../data/zip13.mat')
    train = toy['zip13_train']
    test = toy['zip13_test']
    logregMNIST(train, test)


def testMNIST38():
    '''
     - Load MNIST dataset, characters 3 and 8
     - Train a logistic regression classifyer
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running MNIST38")
    toy = scipy.io.loadmat('../data/zip38.mat')
    train = toy['zip38_train']
    test = toy['zip38_test']
    logregMNIST(train, test)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nLOGREG exercise - MNIST Example")
    print("##########-##########-##########")
    testMNIST13()
    print("##########-##########-##########")
    testMNIST38()
    print("##########-##########-##########")