import operator as op
import math

def bin(value, range, bins):
    '''Divides the interval between range[0] and range[1] into equal sized
    bins, then determines in which of the bins value belongs'''
    bin_size = (range[1] - range[0]) / bins
    return math.floor((value - range[0]) / bin_size)


def binner(range, bins):
	return lambda value: bin(value, range, bins)

# All basis functions should have this form, where `self` will be the agent 
# object. 
def basis(self, observedState):
	pass

# All basis functions must also include a tuple indicating their dimensions
basis.dimensions = (1,)


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

# dimensions of output             map size       x/y  ghosts  pacman
ghostDistance.dimensions = tuple( [   5   ] * (2 * (  3   +  1)) )