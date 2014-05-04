import numpy as np
#import SrcTeam.utils as utils
#import SrcTeam.classifyCapsule as classify
import utils
import classifyCapsule as classify

"""from sklearn import datasets
data = datasets.load_iris().data
classify.k_means(data[0,:], goodSample=data[1:3,:],
            data=data, train=True, plot=True)"""

data = utils.unpickle('SrcTeam/capsuleData/capsule_train')
#classify.k_means(data[0,:], goodSample=data[1:3,:],
#            data=data, train=True, plot=True)
print classify.gaussMixture(data[30,:], goodSample=data[1:10,:],
            data=data, train=False, plot=False)

#print closest((1,1))