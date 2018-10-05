import math
import numpy as np
import matplotlib.pyplot as plt
from imageHelper import imageHelper


def likelihood(data, gmm):
    '''
    Compute the likelihood of each datapoint
    :param data:    Training inputs, #(dims) x #(samples)
    :param gmm:     List of MVND objects
    :return:        Likelihood of each data point
    '''
    likelihood = np.zeros((1, data.shape[0]))
    # Note: For MVGD there will only be 1 item in the list
    for g in gmm:
        likelihood += g.pdf(data)

    return likelihood


def classify(img, mask, sPdf, nPdf, fig="", prior_skin=0.5, prior_nonskin=0.5):
    '''
    :param img:             imageHelper object containing the image to be classified
    :param mask:            imageHelper object containing the ground truth mask
    :param sPdf:            MVND object for the skin class
    :param nPdf:            MVND object for the non-skin class
    :param fig:             Optional figure name
    :param prior_skin:      skin prior, float (0.0-1.0)
    :param prior_nonskin:   nonskin prior, float (0.0-1.0)
    '''
    im_rgb_lin = img.getLinearImage()
    if(type(sPdf) != list):
        sPdf = [sPdf]
    if (type(nPdf) != list):
        nPdf = [nPdf]
    l_skin_rgb = likelihood(im_rgb_lin, sPdf)
    l_nonskin_rgb = likelihood(im_rgb_lin, nPdf)

    testmask = mask.getLinearImageBinary().astype(int)[:,0]
    npixels = len(testmask)

    # TODO: EXERCISE 2 - Error Rate without prior
    fp = None
    fn = None
    totalError = None
    print('----- ----- -----')
    print('Total Error WITHOUT Prior =', totalError)
    print('false positive rate =',fp)
    print('false negative rate =',fn)

    # TODO: EXERCISE 2 - Error Rate with prior
    fp_prior = None
    fn_prior = None
    totalError_prior = None
    print('----- ----- -----')
    print('Total Error WITH Prior =', totalError_prior)
    print('false positive rate =',fp_prior)
    print('false negative rate =',fn_prior)
    print('----- ----- -----')

    # TODO: EXERCISE 2 - Compute and reshape false positive and false negative images
    fpImage = None
    fnImage = None
    fpImagePrior = None
    fnImagePrior = None
    prediction = None
    predictionPrior = None     # Hint: Use or get inspiration from the 'imageHelper' class

    plt.figure(fig)
    plt.subplot2grid((4, 5), (0,0), rowspan=2, colspan=2)
    plt.imshow(img.image);plt.axis('off')
    plt.title('Test image')

    plt.subplot2grid((4, 5), (0,2), rowspan=2, colspan=2)
    plt.imshow(prediction.image, cmap='gray');plt.axis('off')
    plt.title('Skin prediction')

    plt.subplot2grid((4, 5), (2,2), rowspan=2, colspan=2)
    plt.imshow(predictionPrior.image, cmap='gray');plt.axis('off')
    plt.title('Skin prediction PRIOR')

    plt.subplot2grid((4, 5), (2,0), rowspan=2, colspan=2)
    plt.imshow(mask.image, cmap='gray');plt.axis('off')
    plt.title('GT mask')

    plt.subplot(4, 5, 5)
    plt.imshow(fpImage, cmap='gray');plt.axis('off')
    plt.title('FalseNegative')
    plt.subplot(4, 5, 10)
    plt.imshow(fnImage, cmap='gray');plt.axis('off')
    plt.title('FalsePositive')
    plt.subplot(4, 5, 15)
    plt.imshow(fpImagePrior, cmap='gray');plt.axis('off')
    plt.title('FalseNegative PRIOR')
    plt.subplot(4, 5, 20)
    plt.imshow(fnImagePrior, cmap='gray');plt.axis('off')
    plt.title('FalsePositive PRIOR')
