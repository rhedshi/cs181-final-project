import numpy as np
from distanceCalculator import Distancer
from game import Agent
from utils import *

from sklearn.multiclass import *
from sklearn.svm import *
from sklearn import cross_validation
from sklearn import linear_model

import random

# scikit learn classification model objects
ghost_binary_classifier = unpickle('SrcTeam/ghostData/ghost_binary_classifier')
ghost_latent_class_classifier = unpickle('SrcTeam/ghostData/ghost_latent_class_classifier')
# ghost_binary_parameters = unpickle('SrcTeam/ghostData/ghost_binary_parameters')
# ghost_latent_class_parameters = unpickle('SrcTeam/ghostData/ghost_latent_class_parameters')
# ghost_binary_classifier = linear_model.LogisticRegression().set_params(**ghost_binary_parameters)
# ghost_latent_class_classifier = OneVsOneClassifier(LinearSVC()).set_params(**ghost_latent_class_parameters)

# list of class conditional score regression objects
ghost_class_score = [unpickle('SrcTeam/ghostData/ghost_score_' + str(x)) for x in range(4)]

def getLatentClass(feature_vector):
	'''
	feature_vector:		thirteen-dimensional vector as specified in practical description

	return:				integer of the latent class
	'''
	return int(ghost_latent_class_classifier.predict(feature_vector)[0])

def getJuicyScore(latent_class, feature_vector):
	'''
	latent_class:		must be an integer between 0 and 3 inclusive
	feature_vector:		thirteen-dimensional vector as specified in practical description

	return:				float of the juiciness score
	'''
	return float(ghost_class_score[latent_class].predict(feature_vector)[0])

def closestGhost(state, distancer, good=True):
	'''
	state:				the current observedState of the game passed in from studentAgents.py
	distancer:			the self.distancer object passed in from studentAgents.py

	return:				tuple for location of the closest good ghost by default otherwise any closest ghost
	'''
	pacman_position = state.getPacmanPosition()

	ghost_states = state.getGhostStates()
	ghost_feature_vectors = np.array([ghost.getFeatures() for ghost in ghost_states])
	ghost_positions = [ghost.getPosition() for ghost in ghost_states]

	ghost_latent_classes = np.array(map(int,ghost_latent_class_classifier.predict(ghost_feature_vectors)))

	min_distance = np.inf
	min_index = 0
	b = 5 if good else 4
	for i in range(len(ghost_states)):
		if ghost_latent_classes[i] != b:
			distance = distancer.getDistance(pacman_position, ghost_positions[i])
			if distance < min_distance:
				min_distance = distance
				min_index = i
	return ghost_positions[min_index]

def getNearestGoodGhost(state, distancer):
	return closestGhost(state, distancer, True)

def getNearestBadGhost(state, distancer):
	'''
	state:				the current observedState of the game passed in from studentAgents.py
	distancer:			the self.distancer object passed in from studentAgents.py

	return:				tuple for location of the closest bad ghost
	'''
	pacman_position = state.getPacmanPosition()

	ghost_states = state.getGhostStates()
	ghost_feature_vectors = np.array([ghost.getFeatures() for ghost in ghost_states])
	ghost_positions = [ghost.getPosition() for ghost in ghost_states]

	ghost_binary = map(int,ghost_binary_classifier.predict(ghost_feature_vectors))

	min_distance = np.inf
	min_index = random.randint(0,3)
	for i in range(len(ghost_states)):
		if ghost_binary[i] == 1:
			distance = distancer.getDistance(pacman_position, ghost_positions[i])
			if distance < min_distance:
				min_distance = distance
				min_index = i
	return ghost_positions[min_index]