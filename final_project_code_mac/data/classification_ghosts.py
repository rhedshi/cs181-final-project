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

print np.shape(ghost_feature_vector)

methods = [OneVsRestClassifier(LinearSVC()), OneVsOneClassifier(LinearSVC()), linear_model.LogisticRegression()]

k = 10
cv_err = []

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