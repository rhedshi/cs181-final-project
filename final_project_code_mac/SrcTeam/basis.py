import classifyGhosts
import classifyCapsule
import operator as op
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
    if x > range[1]:
    	return bins-1
    elif x < range[0]:
    	return 0

    bin_size = (range[1] - range[0]) / bins
    return math.floor((value - range[0]) / bin_size)

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
def ghostDistance(self,observedState):
	pacmanPosition = observedState.getPacmanPosition()
	ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
	
	bins = 5
	xBinner = binner( (0,observedState.layout.width), bins )	
	yBinner = binner( (0,observedState.layout.height), bins )

	state = (xBinner(pacmanPosition[0]), yBinner(pacmanPosition[1]))
	for gs in ghost_states[0:3]:
		pos = gs.getPosition()
		state += (xBinner(pos[0]),yBinner(pos[1]))
	
	return state

# dimensions of output             map size   x/y  ghosts  pacman
ghostDistance.dimensions = tuple( [   5   ] * (2 * (  3   +  1)) )

# -----------------------------------------------------------------------------

# distances between pacman and the nearest good ghost, the nearest bad ghost,
# and the nearest capsule
def goodBadGhostCapsuleDistances(self,observedState):
	pacmanPosition = observedState.getPacmanPosition()
	goodGhost = classifyGhosts.closestGoodGhost(observedState)
	badGhost = classifyGhosts.closestBadGhost(observedState)
	capsule = classifyCapsule.closest(observedState)

	distBinner = binner( \
		(0,Math.sqrt(observedState.layout.width**2 + observedState.layout.width**2)), \
		goodBadGhostCapsuleDistances.buckets )

	goodGhostDistance = self.distancer(pacmanPosition, goodGhost)
	badGhostDistance = self.distancer(pacmanPosition, badGhost)
	capsuleDistance = self.distancer(pacmanPosition, capsule)
	hasCapsule = observedState.scaredGhostPresent()
	return (goodGhostDistance,badGhostDistance,capsuleDistance,int(hasCapsule))

goodBadGhostCapsuleDistances.buckets = 5

# dimensions of output                            
goodBadGhostCapsuleDistances.dimensions = \
	tuple( [goodBadGhostCapsuleDistances.buckets] * (1   + 1  +  1   ) + [ 2 ] )
	#                     buckets                   good, bad, capsule   eaten?

# -----------------------------------------------------------------------------

# position of the neighborhood around pacman
def localNeighborhood(self, observedState):
	radius = 4
	pacmanPosition = observedState.getPacmanPosition()
	xBinner = ( (pacmanPosition[0]-radius, pacmanPosition[0]+radius), buckets )
	yBinner = ( (pacmanPosition[1]-radius, pacmanPosition[1]+radius), buckets )

	goodGhost = classifyGhosts.closestGoodGhost(observedState)
	badGhost = classifyGhosts.closestBadGhost(observedState)
	capsule = classifyCapsule.closest(observedState)
	hasCapsule = observedState.scaredGhostPresent()

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

	if action == 'good':
		target = classifyGhosts.closestGoodGhost(observedState)
	elif action == 'bad':
		target = classifyGhosts.closestBadGhost(observedState)
	elif action == 'capsule':
		target = classifyCapsule.closest(observedState)
	else:
		raise Exception("Unknown action '%s' passed to action basis" % action)

	# find the direction that moves closes to target
	best_action = Directions.STOP
	best_dist = -np.inf
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