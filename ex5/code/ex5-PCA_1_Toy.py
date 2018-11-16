import sys
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pca import PCA


def toyExample():
    mat = scipy.io.loadmat('../data/toy_data.mat')
    data = mat['toy_data']

    # TODO: Train PCA
    pca = PCA(-1)
    pca.train(data)

    print("Variance of the data")
    # TODO 1.2: Compute data variance to the S vector computed by the PCA
    data_variance = np.var(data,axis=1)
    print(data_variance)
    print(pca.S)
    # TODO 1.3: Compute data variance for the projected data (into 1D) to the S vector computed by the PCA
    Xout = pca.project(data, 1)
    print("Variance of the projected data")
    data_variance = np.var(Xout,axis=1)
    print(data_variance)
    print(pca.S)


    plt.figure()
    plt.title('PCA plot')
    plt.subplot(1,2,1)  # Visualize given data and principal components
    # TODO 1.1: Plot original data (hint, use the plot_pca function
    pca.plot_pca(data)
    plt.subplot(1,2,2)
    # TODO 1.3: Plot data projected into 1 dimension
    pca.plot_pca(Xout)
    plt.show()


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("PCA Toy-example")
    toyExample()
    print("Fertig PCA!")