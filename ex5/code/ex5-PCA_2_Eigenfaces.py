import sys
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import scipy.io
import scipy.spatial
import matplotlib.pyplot as plt
import math
from pca import PCA

# TODO: Implement euclidean distance between two vectors
def euclideanDistance(a, b):
    '''
    :param a: vector
    :param b: vector
    :return: scalar
    '''
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(a, b)]))

# TODO: Implement mahalanobis distance between two vectors
def mahalanobisDistance(a, b, invS):
    '''
    :param a: vector
    :param b: vector
    :param invS: matrix
    :return: scalar
    '''
    return scipy.spatial.distance.mahalanobis(a,b,invS)


def faceRecognition():
    '''
    Train PCA with with 25 components
    Project each face from 'novel' into PCA space to obtain feature coordinates
    Find closest face in 'gallery' according to:
        - Euclidean distance
        - Mahalanobis distance
    Redo with different PCA dimensionality

    What is the effect of reducing the dimensionality?
    What is the effect of different similarity measures?
    '''
    numOfPrincipalComponents = 25
    # TODO: Train a PCA on the provided face images
    pca = PCA(numOfPrincipalComponents)
    (X, data_labels, gall_faces) = data_matrix()
    pca.train(X)
    alphagal = pca.to_pca(X)
    # TODO: Plot the variance of each principal component - use a simple plt.plot()
    plt.plot(np.var(alphagal,1,dtype=np.float64))
    plt.show()
    # TODO: Implement face recognition
    (novel, novel_labels, nov_faces) = load_novel()
    alphanov = pca.to_pca(novel)

    matches_e = []
    matches_m = []

    invS = np.diag(1./pca.S)

    for i in range(alphanov.shape[1]):
        lowest_e = (sys.maxsize, 0)
        lowest_m = (sys.maxsize, 0)

        for j in range(alphagal.shape[1]):
            euclidean = euclideanDistance(alphanov[:,i],alphagal[:,j])

            if euclidean < lowest_e[0]:
                lowest_e = euclidean, j

            maha = mahalanobisDistance(alphanov[:,i],alphagal[:,j], invS)

            if maha < lowest_m[0]:
                lowest_m = maha, j

        matches_e.append((i,lowest_e[1]))
        matches_m.append((i,lowest_m[1]))

    print(matches_e)
    print(matches_m)

    correct_m = 0
    correct_e = 0

    correct_classified = 0
    correct_partner = 0
    wrong_classified = 0
    wrong_partner = 0

    for x in range(len(matches_e)):
        if data_labels[matches_e[x][0]] == novel_labels[matches_e[x][1]]:
            correct_e += 1
            correct_classified = x
            correct_partner = matches_e[x][1]
        else:
            wrong_classified = x
            wrong_partner = matches_e[x][1]
        if data_labels[matches_m[x][0]] == novel_labels[matches_m[x][1]]:
            correct_m += 1


    print("Correct Euclidian Classification in percent: {}".format(correct_e/len(matches_e)*100))
    print("Correct Mahalanobis Classification in percent: {}".format(correct_m / len(matches_m) * 100))

    # TODO: Visualize some of the correctly and wrongly classified images (see example in exercise sheet)

    # Show correct classified
    fig = plt.figure(1)
    columns = 3
    rows = 2
    titles = ("correct test", "wrong test", "projected test", "correct train", "wrong train", "projected train")

    #plt.title(titles[i])

    # correct
    fig.add_subplot(rows, columns, 1)
    plt.imshow(nov_faces.item(correct_classified)[1], cmap='gray')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(gall_faces.item(correct_partner)[1], cmap='gray')
    fig.add_subplot(rows, columns, 3)
    correct_projected = pca.project(novel, numOfPrincipalComponents)[:,correct_classified].reshape(gall_faces.item(correct_partner)[1].shape)
    plt.imshow(correct_projected, cmap='gray')

    # wrong
    fig.add_subplot(rows, columns, 4)
    plt.imshow(nov_faces.item(wrong_classified)[1], cmap='gray')
    fig.add_subplot(rows, columns, 5)
    plt.imshow(gall_faces.item(wrong_partner)[1], cmap='gray')
    fig.add_subplot(rows, columns, 6)
    wrong_projected = pca.project(novel, numOfPrincipalComponents)[:,wrong_classified].reshape(gall_faces.item(wrong_partner)[1].shape)
    plt.imshow(wrong_projected, cmap='gray')

    plt.show()

def load_novel_faces():
    matnov = scipy.io.loadmat('../data/novel.mat')
    nov = matnov['novel'][0]
    return nov

def load_novel():
    matnov = scipy.io.loadmat('../data/novel.mat')
    nov = matnov['novel'][0]

    numOfFaces = nov.shape[0]
    [N, M] = nov.item(0)[1].shape

    print("NumOfFaces in novel dataset", numOfFaces)

    data_matrix = np.zeros((N*M,numOfFaces))
    novID = np.zeros(numOfFaces)
    for i in range(numOfFaces):
        facefirst = nov.item(i)[1]
        faceId = nov.item(i)[0][0]
        data_matrix[:,i] = facefirst.flatten().T
        novID[i] = faceId

    return (data_matrix, novID, nov)

def data_matrix():
    '''
    Hint: In order to do this, you must assemble a data matrix by stacking each image m x n
    into a a column vector mn x 1 and concatenate all column vectors horizontally.
    '''
    matgal = scipy.io.loadmat('../data/gallery.mat')
    gall = matgal['gall'][0]

    numOfFaces = gall.shape[0]
    [N, M] = gall.item(0)[1].shape

    print("NumOfFaces in gallery dataset", numOfFaces)

    data_matrix = np.zeros((N*M,numOfFaces))
    dataID = np.zeros(numOfFaces)
    for i in range(numOfFaces):
        facefirst = gall.item(i)[1]
        faceId = gall.item(i)[0][0]
        data_matrix[:,i] = facefirst.flatten().T
        dataID[i] = faceId

    return (data_matrix, dataID, gall)



def faceLoaderExample():
    '''
    Face loader and visualizer example code
    '''
    matgal = scipy.io.loadmat('../data/gallery.mat')
    gall = matgal['gall'][0]

    numOfFaces = gall.shape[0]
    [N, M] = gall.item(0)[1].shape

    print("NumOfFaces in dataset", numOfFaces)

    # Show first image
    plt.figure(0)
    plt.title('First face')
    n = 0
    facefirst = gall.item(n)[1]
    faceId = gall.item(n)[0][0]
    print('Face got face id: {}'.format(faceId))
    plt.imshow(facefirst, cmap='gray')

    plt.show()




if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("PCA images!")
    #faceLoaderExample()
    faceRecognition()
    print("Fertig PCA!")
