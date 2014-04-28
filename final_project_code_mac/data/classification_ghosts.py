from data_utils import *

import numpy as np
import math

from sklearn.multiclass import *
from sklearn.svm import *
from sklearn import cross_validation
from sklearn import linear_model

ghost_data = csv_to_ndarray('example/ghost_train.csv')

ghost_quadrants = ghost_data[:,0]
ghost_latent_class = ghost_data[:,1]
ghost_score = ghost_data[:,2]
ghost_feature_vector = ghost_data[:,3:]

# print np.shape(ghost_feature_vector)

# ghost_class_0 = ghost_feature_vector[0==ghost_data[:,1]]
# ghost_class_1 = ghost_feature_vector[1==ghost_data[:,1]]
# ghost_class_2 = ghost_feature_vector[2==ghost_data[:,1]]
# ghost_class_3 = ghost_feature_vector[3==ghost_data[:,1]]
# ghost_class_5 = ghost_feature_vector[5==ghost_data[:,1]]

ghost_class_feature_vector = [ghost_feature_vector[x==ghost_data[:,1]] for x in range(4)]

# ghost_score_0 = ghost_score[0==ghost_data[:,1]]
# ghost_score_1 = ghost_score[1==ghost_data[:,1]]
# ghost_score_2 = ghost_score[2==ghost_data[:,1]]
# ghost_score_3 = ghost_score[3==ghost_data[:,1]]
# ghost_score_5 = ghost_score[5==ghost_data[:,1]]

ghost_class_score = [ghost_score[x==ghost_data[:,1]] for x in range(4)]

k = 10
cv_err = []

# Latent Class Ghost Classifier
'''
methods = [OneVsRestClassifier(LinearSVC()), OneVsOneClassifier(LinearSVC()), linear_model.LogisticRegression()]

X = ghost_feature_vector
y = ghost_latent_class

kf = cross_validation.KFold(len(y),n_folds=k,shuffle=True)

for method in methods:
	print str(method)
	print "k-fold cross-validation with %d folds" % k
	for train,tests in kf:

	    X_train, y_train, X_test, y_test = X[train], y[train], X[tests], y[tests]

	    result = method.fit(X_train,y_train)
	    preds = result.predict(X_test)

	    cv_err.append(1 - float(np.sum(preds == y_test)) / len(y_test))
	    print "Err on withheld data: %f" % cv_err[-1]

	# calculate mean, std. across folds
	cv_err_mean, cv_err_std = np.mean(cv_err), np.std(cv_err)

	print
	print "Avg. Err: %f" % cv_err_mean
	print "Std. Err: %f" % cv_err_std

OneVsOne = OneVsOneClassifier(LinearSVC()).fit(X,y)
pickle(OneVsOne,'ghost_latent_class_classifier.pkl')
'''

# Latent Class Conditional Score Regression

for i in range(4):
	print "Linear Regression on Scores for Latent Class " + str(i)
	print "k-fold cross-validation with %d folds" % k

	X = ghost_class_feature_vector[i]
	y = ghost_class_score[i]

	kf = cross_validation.KFold(len(y),n_folds=k,shuffle=True)

	for train,tests in kf:

	    X_train, y_train, X_test, y_test = X[train], y[train], X[tests], y[tests]

	    result = linear_model.BayesianRidge().fit(X_train,y_train)
	    preds = result.predict(X_test)

	    cv_err.append(result.score(X_test,y_test))
	    print "Err on withheld data: %f" % cv_err[-1]

	# calculate mean, std. across folds
	cv_err_mean, cv_err_std = np.mean(cv_err), np.std(cv_err)

	print
	print "Avg. Err: %f" % cv_err_mean
	print "Std. Err: %f" % cv_err_std

	model = linear_model.BayesianRidge().fit(X,y)
	print model.score(X,y)
	pickle(model,'ghost_score_' + str(i) + '.pkl')
