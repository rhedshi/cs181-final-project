from data_utils import *

import numpy as np
import math

from sklearn.multiclass import *
from sklearn.svm import *
from sklearn import cross_validation
from sklearn import linear_model

ghost_binary_classifier = unpickle('ghost_binary_classifier')
ghost_latent_class_classifier = unpickle('ghost_latent_class_classifier')

ghost_binary_parameters = ghost_binary_classifier.get_params()
ghost_latent_class_parameters = ghost_latent_class_classifier.get_params()

pickle(ghost_binary_parameters,'ghost_binary_parameters')
pickle(ghost_latent_class_parameters,'ghost_latent_class_parameters')