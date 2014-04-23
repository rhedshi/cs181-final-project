import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
#import observedState as state
import utils

def isGood(sample):
    "returns True if the sample capsule is a non-placebo"

    goodSamples = state.getGoodCapsuleExamples()

    if(k_means(sample, goodSamples) > 0.5):
        return True

    return False

def k_means(testData, goodSample,
            data=None, train=True, plot=False):
    if train==True:
        n_clusters = 3
        est = KMeans(n_clusters)
        est.fit(data)
        centers = est.cluster_centers_
        utils.pickle(est, 'data/capsule_k_means')
    else:
        est = utils.unpickle('data/capsule_k_means')

    numMatch = 0.0
    numGood = goodSample.shape[0]

    #sampleLabel = clusterLabel(centers, sample)
    testLabel = est.predict(testData)
    for i in range(numGood):
        if est.predict(goodSample[i,:]) == testLabel:
            numMatch += 1

    if plot==True:
        fig = pl.figure()
        pl.clf()
        ax = Axes3D(fig)
        labels = est.labels_
        ax.scatter(data[:,0],data[:,1],data[:,2],c=labels.astype(np.float))
        pl.show()

    return float(numMatch) / numGood

def gaussMixture(sample, goodSample,
                 data=None, train=True, plot=False):
    if train==True:
        n_classes = 3
        covar_type = 'full'
        classifier = GMM(n_components=n_classes, covariance_type=covar_type)
        classifier.fit(data)
        utils.pickle(classifier, 'data/capsule_gauss')
    else:
        utils.classifier = unpickle('data/capsule_gauss')

    numMatch = 0.0
    numGood = goodSample.shape[0]

    sampleLabel = classifier.predict(sample)
    for i in range(numGood):
        if classifier.predict(goodSample[i,:]) == sampleLabel:
            numMatch += 1

    if plot==True:
        fig = pl.figure()
        pl.clf()
        ax = Axes3D(fig)
        labels = est.predict(data)
        ax.scatter(data[:,0],data[:,1],data[:,2],c=labels.astype(np.float))
        pl.show()

    return float(numMatch) / numGood

"""def clusterLabel(centers, sample):
    dists = np.zeros(centers.shape[0])
    for i in range(centers.shape[0]):
        dists[i] = dist(sample, centers)
    return argmin

def dist(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)"""

