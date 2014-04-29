from data_utils import *
from sklearn import *

import numpy as np

ghost_data = csv_to_ndarray('ghost_train.csv')

ghost_quadrants = ghost_data[:,0]
ghost_latent_class = ghost_data[:,1]
ghost_score = ghost_data[:,2]
ghost_feature_vector = ghost_data[:,3:]

good_ghosts_feature_vector = ghost_feature_vector[5!=ghost_data[:,1]]
good_ghosts_score = ghost_score[5!=ghost_data[:,1]]

ghost_latent_class_classifier = unpickle('ghost_latent_class_classifier')
ghost_class_score = [unpickle('ghost_score_' + str(x%4)) for x in range(6)]

pred_class = ghost_latent_class_classifier.predict(good_ghosts_feature_vector)

# print 5 in pred_class

pred_score = np.array([ghost_class_score[int(x)].predict(good_ghosts_feature_vector[int(i[0])]) for i,x in np.ndenumerate(pred_class)])

print 1 - float(np.sum(np.square(pred_score - good_ghosts_score))) / len(good_ghosts_score)