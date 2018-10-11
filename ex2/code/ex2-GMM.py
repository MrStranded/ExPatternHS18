import sys, os, math, time
import random
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from imageHelper import imageHelper
from myMVND import MVND
from classifyHelper import classify

dataPath = '../data/'

def gmm_draw(gmm, data, plotname=''):
    '''
    gmm helper function to visualize cluster assignment of data
    :param gmm:         list of MVND objects
    :param data:        Training inputs, #(dims) x #(samples)
    :param plotname:    Optional figure name
    '''
    plt.figure(plotname)
    K = len(gmm)
    N = data.shape[1]
    dists = np.zeros((K,N))
    for k in range(0,K):
        d = data - np.transpose(np.kron(np.ones((N, 1)), gmm[k].mean))
        dists[k,:] = np.sum(np.multiply(np.matmul(np.linalg.inv(np.matrix(gmm[k].cov)), np.matrix(d)), np.matrix(d)), axis=0)
    comp = np.argmin(dists, axis=0)

    # plot the input data
    ax = plt.gca()
    ax.axis('equal')
    for (k, g) in enumerate(gmm):
        indexes = np.where(comp == k)[0]
        kdata = data[:, indexes]
        g.data = kdata
        ax.scatter(kdata[0, :], kdata[1, :])

        [_, L, V] = scipy.linalg.svd(g.cov, full_matrices=False)
        phi = math.acos(V[0, 0])
        if float(V[1,0]) < 0.0:
            phi = 2*math.pi - phi
        phi = 360-(phi*180/math.pi)
        center = np.array(g.mean).reshape(1,-1)

        d1 = 2*np.sqrt(L[0])
        d2 = 2*np.sqrt(L[1])
        ax.add_patch(Ellipse(np.transpose(center), d1, d2, phi, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1, fill=False))
        plt.plot(center[0,0], center[0,1], 'kx')


def gmm_em(data, K, iter, plot=False):
    '''
    EM-algorithm for Gaussian Mixture Models
    Usage: gmm = gmm_em(data, K, iter)
    :param data:    Training inputs, #(dims) x #(samples)
    :param K:       Number of GMM components, integer (>=1)
    :param iter:    Number of iterations, integer (>=0)
    :param plot:    Enable/disable debugging plotting
    :return:        List of objects holding the GMM parameters.
                    Use gmm[i].mean, gmm[i].cov, gmm[i].p
    '''
    eps = sys.float_info.epsilon
    [d, N] = data.shape
    gmm = []
    list1 = np.zeros((2, 0))
    list2 = np.zeros((2, 0))
    list3 = np.zeros((2, 0))

    for n in range(N):
        r = random.randint(1, 3)
        if r == 1:
            list1 = np.concatenate((list1, np.array([data[:, n]]).T), axis=1)
        elif r == 2:
            list2 = np.concatenate((list2, np.array([data[:, n]]).T), axis=1)
        elif r == 3:
            list3 = np.concatenate((list3, np.array([data[:, n]]).T), axis=1)


    print(list1)
    gmm = [MVND(list1), MVND(list2), MVND(list3)]

    # TODO: EXERCISE 2 - Implement E and M step of GMM algorithm
    # Hint - first randomly assign a cluster to each sample - check
    # Hint - then iteratively update mean, cov and p value of each cluster via EM
    # Hint - use the gmm_draw() function to visualize each step
    gmm_draw(gmm,data,"test")


    plt.show()
    return gmm


def gmmToyExample():
    '''
    GMM toy example - load toyexample data and visualize cluster assignment of each datapoint
    '''
    gmmdata = scipy.io.loadmat(os.path.join(dataPath,'gmmdata.mat'))['gmmdata']
    gmm_em(gmmdata, 3, 20, plot=True)


def gmmSkinDetection():
    '''
    Skin detection - train a GMM for both skin and non-skin.
    Classify the test and training image using the classify helper function.
    Note that the "mask" binary images are used as the ground truth.
    '''
    K = 3
    iter = 50
    sdata = scipy.io.loadmat(os.path.join(dataPath,'skin.mat'))['sdata']
    ndata = scipy.io.loadmat(os.path.join(dataPath,'nonskin.mat'))['ndata']
    gmms = gmm_em(sdata, K, iter)
    gmmn = gmm_em(ndata, K, iter)

    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'image.png'))
    print("TRAINING DATA")
    classify(testimageObj, testmaskObj, gmms, gmmn, "training")

    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask-test.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'test.png'))
    print("TEST DATA")
    classify(testimageObj, testmaskObj, gmms, gmmn, "test")


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nMVND exercise - Toy example")
    print("##########-##########-##########")
    gmmToyExample()
    print("\nMVND exercise - Skin detection")
    print("##########-##########-##########")
    gmmSkinDetection()
    print("##########-##########-##########")
