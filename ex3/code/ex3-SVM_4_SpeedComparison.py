import sys
import time
import numpy as np
import scipy.io
from svm import SVM


def svmSpeedComparison():
    '''
    Train a linear SVM and a Kernel SVM with a linear kernel
    Time the classification functions
    Note the average time over 1000 runs for both classifiers
    '''
    numOfRuns = 1000
    print("Speed comparison")

    toy = scipy.io.loadmat('../data/toy.mat')
    toy_train = toy['toy_train']
    toy_test = toy['toy_test']

    toy_train_label = np.transpose(toy_train[0,:].astype(np.double)[:,None])
    toy_train_x     = toy_train[1:3,:].astype(np.double)

    toy_test_x     = toy_test[1:3,:].astype(np.double)

    # training the SVMs
    svm_lin = SVM(None)
    svm_lin.train(toy_train_x, toy_train_label)

    svm_ker = SVM(None)
    svm_ker.train(toy_train_x,toy_train_label,'linear',0)

    # TODO: Compute the average classification time of both the linear and the kernel SVM (with a linear kernel)
    sum_linear = 0
    sum_kernel = 0
    for i in range(numOfRuns):
        t = time.time()
        svm_lin.classifyLinear(toy_test_x)
        sum_linear += (time.time() - t)

        t = time.time()
        svm_ker.classifyKernel(toy_test_x)
        sum_kernel += (time.time() - t)
    result_linear = sum_linear / numOfRuns
    result_kernel = sum_kernel / numOfRuns

    print('Linear SVM timing: \n {:.10f} over {} runs'.format(result_linear, numOfRuns))
    print('SVM with linear kernel timing: \n {:.10f} over {} runs'.format(result_kernel, numOfRuns))
    print('Linear is {} times faster'.format(result_kernel/result_linear))
    # kernel is about 10 times faster
    # maybe because in kernel case, only support vectors are considered,
    # whereas in linear case, we go through each data point


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nSVM exercise - Speed comparison")
    print("##########-##########-##########")
    svmSpeedComparison()
    print("##########-##########-##########")
