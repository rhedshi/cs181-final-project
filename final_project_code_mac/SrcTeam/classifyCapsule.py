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

def closest():
    "returns a tuple, the position of the nearest non-placebo capsule"
    agent = state.getPacmanState().getPosition()
    capsules = state.getCapsuleData()
    goodCapsules = [i for i in capsules if isGood(i)]
    minCapsule = goodCapsules[0]
    minDist = dist(goodCapsule[1],agent)
    for caps in goodCapsules:
        if dist(caps,agent)<minDist:
            minDist = dist(caps,agent)
            minCapsule = caps

    return minCapsule

def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def k_means(testX, goodSample,
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
    testLabel = est.predict(testX)
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

def gaussMixture(testX, goodSample,
                 data=None, train=True, plot=False):
    if train==True:
        n_classes = 3
        covar_type = 'full'
        est = GMM(n_components=n_classes, covariance_type=covar_type)
        est.fit(data)
        utils.pickle(classifier, 'data/capsule_gauss')
    else:
        est = utils.unpickle('data/capsule_gauss')

    numMatch = 0.0
    numGood = goodSample.shape[0]

    testData = np.reshape(testX,(1,testX.size))
    predLabel = est.predict(testData)
    for i in range(numGood):
        if est.predict(goodSample[i:i+1,:]) == predLabel:
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

