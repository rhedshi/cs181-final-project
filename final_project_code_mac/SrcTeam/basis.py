import classifyGhosts
import classifyCapsule
import operator as op
import numpy as np
import math
from game import Actions
from game import Directions


def bin(value, range, bins):
    '''Divides the interval between range[0] and range[1] into equal sized
    bins, then determines in which of the bins value belongs'''
    bin_size = (range[1] - range[0]) / bins
    return math.floor((value - range[0]) / bin_size)

def xbin(value, range, bins):
    '''Divides the interval between range[0] and range[1] into equal sized
    bins, then determines in which of the bins value belongs. If value is 
    outside of range, returns the largest or smallest bin (respectively)'''
    if value >= range[1]:
    	return bins-1
    elif value <= range[0]:
    	return 0

    bin_size = (range[1] - range[0]) / bins
    return int(min(max(math.floor((value - range[0]) / bin_size),0),bins-1))

def binner(range, bins):
	return lambda value: bin(value, range, bins)

def xbinner(range, bins):
	return lambda value: xbin(value, range, bins)


# -----------------------------------------------------------------------------

# All basis functions should have this form, where `self` will be the agent 
# object. 
def basis(self, observedState):
	pass

# All basis functions must also include a tuple indicating their dimensions
basis.dimensions = (1,)

# -----------------------------------------------------------------------------


# gives pacman's position, and the position of each of the ghosts
def ghostPosition(self,observedState):
	pacmanPosition = observedState.getPacmanPosition()
	ghost_positions = [ classifyGhosts.getNearestGoodGhost(observedState, self.distancer), classifyGhosts.getNearestBadGhost(observedState, self.distancer)]
	
	bins = 5
	xBinner = xbinner( (0,observedState.layout.width), bins )	
	yBinner = xbinner( (0,observedState.layout.height), bins )

	state = (xBinner(pacmanPosition[0]), yBinner(pacmanPosition[1]))
	for pos in ghost_positions:
		state += (xBinner(pos[0]),yBinner(pos[1]))
	
	return state

# dimensions of output             map size   x/y  ghosts  pacman
ghostPosition.dimensions = tuple( [   5   ] * (2 * (  2   +  1)) )

# -----------------------------------------------------------------------------

# distances between pacman and the nearest good ghost, the nearest bad ghost,
# and the nearest capsule
def goodBadGhostCapsuleDistances(self,observedState):
	pacmanPosition = observedState.getPacmanPosition()
	goodGhost = classifyGhosts.getNearestGoodGhost(observedState, self.distancer)
	badGhost = classifyGhosts.getNearestBadGhost(observedState, self.distancer)
	capsule = classifyCapsule.closest(observedState, self.distancer)

	distBinner = xbinner( \
		(0,math.sqrt(observedState.layout.width**2 + observedState.layout.width**2)), \
		goodBadGhostCapsuleDistances.buckets )

	goodGhostDistance = distBinner(self.distancer.getDistance(pacmanPosition, goodGhost))
	badGhostDistance = distBinner(self.distancer.getDistance(pacmanPosition, badGhost))
	capsuleDistance = distBinner(self.distancer.getDistance(pacmanPosition, capsule))
	hasCapsule = observedState.scaredGhostPresent()
	return (goodGhostDistance,badGhostDistance,capsuleDistance,int(hasCapsule))

goodBadGhostCapsuleDistances.buckets = 10

# dimensions of output                            
goodBadGhostCapsuleDistances.dimensions = \
	tuple( [goodBadGhostCapsuleDistances.buckets] * (1   + 1  +  1   ) + [ 2 ] )
	#                     buckets                   good, bad, capsule   eaten?


# -----------------------------------------------------------------------------

# position of the neighborhood around pacman
def localNeighborhood(self, observedState):
	radius = 4
	pacmanPosition = observedState.getPacmanPosition()

	xBinner = xbinner( (pacmanPosition[0]-radius, pacmanPosition[0]+radius), localNeighborhood.buckets )
	yBinner = xbinner( (pacmanPosition[1]-radius, pacmanPosition[1]+radius), localNeighborhood.buckets )

	goodGhost = classifyGhosts.getNearestGoodGhost(observedState, self.distancer)
	badGhost = classifyGhosts.getNearestBadGhost(observedState, self.distancer)
	capsule = classifyCapsule.closest(observedState, self.distancer)
	hasCapsule = observedState.scaredGhostPresent()


	print pacmanPosition, goodGhost, badGhost, capsule, hasCapsule

	# bin the x/y position for each object of interest 
	b = []
	for p in [goodGhost, badGhost, capsule]:
		b += [xBinner(p[0]), yBinner(p[1])]
	return tuple(b + [int(hasCapsule)])

localNeighborhood.buckets = 4

# dimensions of output                           
localNeighborhood.dimensions = tuple( [localNeighborhood.buckets] * (2 * (1   + 1  +  1   )) + [ 2 ] )
                               #                 buckets            x/y  good  bad  capsule    eaten

# ============================================================================

def followActionBasis(self, observedState, action):
	allowedDirections = observedState.getLegalPacmanActions()
	pacmanPosition = observedState.getPacmanPosition()

	if action == 'good':
		target = classifyGhosts.getNearestGoodGhost(observedState, self.distancer)
	elif action == 'bad':
		target = classifyGhosts.getNearestBadGhost(observedState, self.distancer)
	elif action == 'capsule':
		target = classifyCapsule.closest(observedState, self.distancer)
	else:
		raise Exception("Unknown action '%s' passed to action basis" % action)

	# find the direction that moves closes to target
	best_action = Directions.STOP
	best_dist = np.inf
	for la in allowedDirections:
		if la == Directions.STOP:
			continue
		successor_pos = Actions.getSuccessor(pacmanPosition,la)
		new_dist = self.distancer.getDistance(successor_pos,target)
		if new_dist < best_dist:
			best_action = la
			best_dist = new_dist
	return best_action

def followActionBasisAllowedActions(self, observedState):
	return followActionBasis.allActions

followActionBasis.allowedActions = followActionBasisAllowedActions
followActionBasis.allActions = ['good', 'bad', 'capsule']

# -----------------------------------------------------------------------------

def simpleActionBasis(self, observedState, action):
	return action

def simpleActionBasisAllowedActions(self, observedState):
	return observedState.getLegalPacmanActions()

simpleActionBasis.allowedActions = simpleActionBasisAllowedActions
simpleActionBasis.allActions = Actions._directions.keys()

# ============================================================================

initializers = {}

def goodBadFollowInitializer(learner):
	actionCodes = { action: i for (i, action) in enumerate(followActionBasis.allActions) }
	hasCapsuleIndex = -1
	for s in np.ndindex(goodBadGhostCapsuleDistances.dimensions):
		learner.Q[s + (actionCodes['bad'],)] = -1200 if s[hasCapsuleIndex] == 0 else 0
		learner.Q[s + (actionCodes['capsule'],)] = -100 if s[hasCapsuleIndex] == 1 else 0 
	

initializers['goodBadGhostCapsuleDistances','followActionBasis'] = goodBadFollowInitializer

def localNeighborhoodInitializer(learner):
	actionCodes = { action: i for (i, action) in enumerate(simpleActionBasis.allActions) }
	
	badGhostXIndex = 2
	badGhostYIndex = 3
	centerPos = localNeighborhood.buckets / 2.0
	maxPos = localNeighborhood.buckets - 1
	minPos = 0
	topPos,botPos = maxPos, minPos
	lefPos,rigPos = minPos, maxPos
	hasCapsuleIndex = -1
	for s in np.ndindex(localNeighborhood.dimensions):
		if s[badGhostYIndex] >= centerPos and s[badGhostYIndex] < topPos and s[hasCapsuleIndex] == 0: learner.Q[s + (actionCodes['North'],)] = -1200
		if s[badGhostYIndex] <  centerPos and s[badGhostYIndex] > botPos and s[hasCapsuleIndex] == 0: learner.Q[s + (actionCodes['South'],)] = -1200
		if s[badGhostXIndex] <  centerPos and s[badGhostXIndex] > lefPos and s[hasCapsuleIndex] == 0: learner.Q[s + (actionCodes['West'],)] = -1200
		if s[badGhostXIndex] >= centerPos and s[badGhostXIndex] < rigPos and s[hasCapsuleIndex] == 0: learner.Q[s + (actionCodes['East'],)] = -1200

		if s[badGhostYIndex] > botPos and s[badGhostYIndex] < topPos and s[hasCapsuleIndex] == 0: learner.Q[s + (actionCodes['Stop'],)] = -1200
		if s[badGhostXIndex] < rigPos and s[badGhostYIndex] > lefPos and s[hasCapsuleIndex] == 0: learner.Q[s + (actionCodes['Stop'],)] = -1200

initializers['localNeighborhood','simpleActionBasis'] = localNeighborhoodInitializer
