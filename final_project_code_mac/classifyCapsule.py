import numpy as np
from sklearn.cluster import KMeans
import observedState as state

def isGood(sample):
    "returns True if the sample capsule is a non-placebo"

    goodSamples = state.getGoodCapsuleExamples()

    if(k_means(sample, goodSamples) > 0.5):
        return True

    return False

def k_means(sample, goodSample=state.getGoodCapsuleExamples(),
            data=None, train=True, plot=False):
    if train==true:
        n_clusters = 3
        est = KMeans(n_clusters)
        est.fit(data)
        centers = est.cluster_centers_
        pickle(centers, 'capsule_k_means')
    else:
        centers = unpickle('capsule_k_means')

    numMatch = 0.0
    numGood = goodSample.shape[0]

    sampleLabel = clusterLabel(centers, sample)
    for i in range(numGood):
        if clusterLabel(centers,goodSample[i]) == sampleLabel:
            numMatch += 1

    return float(numMatch) / numGood

def clusterLabel(centers, sample):
    dists = np.zeros(centers.shape[0])
    for i in range(centers.shape[0]):
        dists[i] = dist(sample, centers)
    return argmin

def dist(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def unpickle(file):
    """Loads and returns a pickled data structure in the given `file` name
    Example usage:
        data = unpickle('output/U_20_std')
    """
    fo = open(file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data

def pickle(data, file):
    """Dumps data to a file
    Example usage:
        pickle(U, 'output/U_20_std')
    """
    fo = open(file,'wb')
    cPickle.dump(data,fo)
    fo.close()