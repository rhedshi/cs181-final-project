import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
# import pylab as pl
# from mpl_toolkits.mplot3d import Axes3D
from distanceCalculator import Distancer
import utils

def isGood(state, sample):
    "returns True if the sample capsule is a non-placebo"

    goodSamples = state.getGoodCapsuleExamples()

    if(gaussMixture(sample, goodSamples) > 0.5):
        return True

    return False

def closest(state, distancer, agent=None):
    "returns a tuple, the position of the nearest non-placebo capsule"

    #Get Pacman's position
    if agent==None:
        agent = state.getPacmanPosition()
    capsules = state.getCapsuleData()
    #Sort out the good capsules
    goodCapsules = [i for i in capsules if isGood(state, i[1])]


    minCapsule = goodCapsules[0]
    minDist = distancer.getDistance(minCapsule[0],agent)

    for caps in goodCapsules:
        if distancer.getDistance(caps[0],agent) < minDist:
            minDist = distancer.getDistance(caps[0],agent)
            minCapsule = caps

    return minCapsule[0]

def k_means(testX, goodSample,
            data=None, train=False, plot=False):
    if train==True:
        n_clusters = 3
        est = KMeans(n_clusters)
        est.fit(data)
        centers = est.cluster_centers_
        utils.pickle(est, 'SrcTeam/clusterData/capsule_k_means')
    else:
        est = utils.unpickle('SrcTeam/clusterData/capsule_k_means')

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
                 data=None, train=False, plot=False):
    if train==True:
        n_classes = 3
        covar_type = 'full'
        est = GMM(n_components=n_classes, covariance_type=covar_type)
        est.fit(data)
        utils.pickle(est, 'SrcTeam/clusterData/capsule_gauss')
    else:
        est = utils.unpickle('SrcTeam/clusterData/capsule_gauss')

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

def gaussMixtureOld(testX, goodSample,
                 data=None, train=False, plot=False):
    if train==True:
        n_classes = 3
        covar_type = 'full'
        est = GMM(n_components=n_classes, cvtype=covar_type)
        est.fit(data)
        utils.pickle(est, 'SrcTeam/clusterData/capsule_gauss_old')
    else:
        est = utils.unpickle('SrcTeam/clusterData/capsule_gauss_old')

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