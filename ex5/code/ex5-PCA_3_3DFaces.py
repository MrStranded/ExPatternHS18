import sys
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from pca import PCA
from meshHelper import initializeFaces, renderFace


def renderRandomFace(faces, pca, num):
    '''
    Render random faces
    :param faces: triangulatedSurfaces object (see class in meshHelper)
    :param pca: trained pca
    :param num: number of random faces to show
    '''
    # TODO 3.2: Implement missing functionality
    print('Render {} random faces'.format(num))
    size = faces.meshes.shape[1]
    for i in range(0,num):
        alpha = (np.random.rand(size)*2-np.ones(size))
        alpha = alpha * pca.S
        alpha.reshape((size,1))
        face = pca.from_pca(alpha)
        # TODO: Render face with the renderFace function
        renderFace(face, faces.triangulation,name="random" + str(i))


def lowRankApproximation(faces, pca):
    '''
    Loads a face and renders different low dimensional approximations of it
    :param faces: triangulatedSurfaces object (see class in meshHelper)
    :param pca: trained pca
    '''
    # TODO 3.3: Implement missing functionality
    print('3D face - low rank approximation')
    face = faces.getMesh(2)[:,None]
    for i in range(1, len(pca.S)):
        print(' - Approximation to : {} components'.format(i))
        projection = pca.project(face, i)
        # TODO: Render face with the renderFace function
        renderFace(projection, faces.triangulation,name="PCA " + str(i))


def faces3DExample():
    '''
     - First initialize all faces (load 11 .ply meshes) - this function reshapes the 3D point coordinates into a single vector format
     - Render a face from the initialized faces
     - Instanciate and train PCA
     - Render random faces from the trained pca
     - Project a face into different lower dimensional pca spaces
    '''
    # Note: that faces is an object of the 'triangulatedSurfaces' class found in meshHelper
    # The triangulation needed for the renderer is stored in 'faces.triangulation'
    # All the 3D face meshes in vectorized format is stored in 'faces.meshes'
    # Use the helper function 'faces.getMesh(2)' to obtain one of the vectorized 3D faces
    faces = initializeFaces(pathToData = '../data/face-data/')

    renderFace(faces.getMesh(2), faces.triangulation, name="Vetter")
    # TODO 3.1: Train PCA with the 3D face dataset
    pca = PCA()
    pca.train(faces.meshes)

    renderRandomFace(faces, pca, 3)

    lowRankApproximation(faces, pca)


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("PCA 3D faces!")
    faces3DExample()
    print("Fertig PCA!")
