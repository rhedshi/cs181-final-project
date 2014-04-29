import numpy as np
from observedState import ObservedState
from distanceCalculator import Distancer
from game import Agent
from utils import *

# scikit learn classification model objects
ghost_binary_classifier = unpickle('data/ghost_binary_classifier')
ghost_latent_class_classifier = unpickle('data/ghost_latent_class_classifier')

# list of class conditional score regression objects
ghost_class_score = [unpickle('../data/ghost_score_' + str(x)) for x in range(4)]

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

def closestGoodGhost(observedState=ObservedState):
	'''
	observedState:		object specified in the observedState.py file

	return:				index of the closest good ghost
	'''
	pacman_position = observedState.getPacmanPosition()

	ghost_states = observedState.getGhostStates()
	ghost_feature_vectors = ghost_states.getFeatures()
	ghost_positions = ghost_states.getPosition()

	ghost_latent_classes = np.array(map(int,ghost_latent_class_classifier.predict(ghost_feature_vectors)))

	min_distance = np.inf
	min_index = 0
	for i in range(len(ghost_states)):
		if ghost_latent_classes[i] != 5:
			distance = Distancer.getDistance(pacman_position, ghost_positions[i])
			if distance < min_distance:
				min_distance = distance
				min_index = i
	return min_index