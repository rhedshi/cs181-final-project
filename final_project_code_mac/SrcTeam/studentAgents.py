from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
from learners import model_based, model_free, td_value
import basis
import classifyCapsule as capsule

class BaseStudentAgent(object):
    """Superclass of agents students will write"""

    def registerInitialState(self, gameState):
        """Initializes some helper modules"""
        import __main__
        self.display = __main__._display
        self.distancer = Distancer(gameState.data.layout, False)
        self.firstMove = True

    def observationFunction(self, gameState):
        """ maps true state to observed state """
        return ObservedState(gameState)

    def getAction(self, observedState):
        """ returns action chosen by agent"""
        return self.chooseAction(observedState)

    def chooseAction(self, observedState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


## Below is the class students need to rename and modify

class SrcTeamAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """

    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        pass # you probably won't need this, but just in case

    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(SrcTeamAgent, self).registerInitialState(gameState)

        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")

        # get all the allowed actions, encode them as learners
        self.actions = gameState.getLegalPacmanActions()
        self.actionCodes = { action: i for (i, action) in enumerate(self.actions) }

        # ---------------------------------------------------------------------
        # pick a basis function here
        self.basis = basis.ghostDistance
        # ---------------------------------------------------------------------

        # remember basis function dimensions
        self.basis_dimensions = self.basis.dimensions

        # ---------------------------------------------------------------------
        # pick a learner here
        self.learner = model_free.ModelFreeLearner(self.basis_dimensions, self.actionCodes.values())
        # self.learner = td_value.TDValueLearner(self.basis_dimensions, self.actionCodes.values())
        # ---------------------------------------------------------------------

        # initialize score
        self.score = 0

    def chooseAction(self, observedState):
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.

        This silly pacman agent will move away from the ghost that it is closest
        to. This is not a very good strategy, and completely ignores the features of
        the ghosts and the capsules; it is just designed to give you an example.
        """

        # calculate reward (score delta) for last action
        current_score = observedState.score
        last_score = self.score
        reward = last_score - current_score

        # pass reward to learner
        self.learner.reward_callback(reward)

        # apply basis function to calculate new state
        state = self.basis(self,observedState)

        # ask learner to plan new state
        allowed_action_codes = [self.actionCodes[a] for a in observedState.getLegalPacmanActions()]
        action_code = self.learner.action_callback(state,allowed_action_codes)

        # update score
        self.last_score = current_score

        # take action
        return self.actions[action_code]


class SmartAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """

    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        pass # you probably won't need this, but just in case

    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(SrcTeamAgent, self).registerInitialState(gameState)

    def chooseAction(self, observedState):
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.

        This silly pacman agent will move away from the ghost that it is closest
        to. This is not a very good strategy, and completely ignores the features of
        the ghosts and the capsules; it is just designed to give you an example.
        """

        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition())
                              for gs in ghost_states])
        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]
        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = -np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())
            if new_dist > best_dist:
                best_action = la
                best_dist = new_dist
        return best_action