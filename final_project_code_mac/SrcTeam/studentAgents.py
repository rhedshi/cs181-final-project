from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
from learners import model_based, model_free, td_value
import basis
import classifyCapsule as capsule
import mapHelper as mapH
import classifyGhosts as ghosts
import os.path
import utils
import random

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

# =============================================================================

class ActionBasisAgent(BaseStudentAgent):
    """
    An agent that does reinforcement learning with a configurable learning
    method (Q-learning or TD-value learning), basis function, and action-basis
    function (see basis.py). Basis functions map the observedState to a tuple of
    integers representing the current state of the MDP, while action basis
    functions allow the recommended action from the learner to be mapped to a
    direction. This allows e.g. for the learner to decide whether to follow the
    good ghost, the bad ghost, or a capsule---rather than simply deciding
    whether to go up/down/left/right.
    """

    # ---------------------------------------------------------------------
    # pick an action basis function here
    actionBasis = basis.followActionBasis
    # actionBasis = basis.simpleActionBasis
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # pick a basis function here
    # basis = basis.ghostDistance
    basis = basis.goodBadGhostCapsuleDistances
    # basis = basis.localNeighborhood
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # pick a learner here
    learner_class = model_free.ModelFreeLearner
    # learner_class = td_value.TDValueLearner
    # ---------------------------------------------------------------------

    use_initializer = False

    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        self.learn_file = (kwargs['file'] if 'file' in kwargs \
            else 'SrcTeam/data/learn_'+self.__class__.__name__)
        self.save_every = (kwargs['save_every'] if 'save_every' in kwargs else 100)
        self.restart_learning = (kwargs['restart'] if 'restart' in kwargs else False)
        self.chatter = (kwargs['chatter'] if 'chatter' in kwargs else False)

    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(ActionBasisAgent, self).registerInitialState(gameState)

        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")



        # get all the allowed actions, encode them for learners
        self.actions = self.actionBasis.allActions[:]
        self.actionCodes = { action: i for (i, action) in enumerate(self.actions) }

        # remember basis function dimensions
        self.basis_dimensions = self.basis.dimensions

        # try to load the learner from a file
        if(self.learn_file and os.path.isfile(self.learn_file) and not self.restart_learning):
            self.learner = utils.unpickle(self.learn_file)
            self.learner.reset()
        else:
            self.learner = self.learner_class(self.basis_dimensions, self.actionCodes.values())
            if(self.use_initializer):
                if (self.basis.__name__, self.actionBasis.__name__) in basis.initializers:
                    basis.initializers[(self.basis.__name__, self.actionBasis.__name__)](self.learner)
                    print "Initialized successfully"
                else:
                    print "No initializer found"
                    raise "Whoops"

        # initialize score
        self.score = 0

        # count number of actions taken
        self.action_count = 0

    def chooseAction(self, observedState):
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.
        """

        # calculate reward (score delta) for last action
        current_score = observedState.score
        last_score = self.score
        reward = current_score - last_score
        if self.chatter: print reward

        # pass reward to learner
        self.learner.reward_callback(reward)

        # apply basis function to calculate new state
        state = self.basis(observedState)

        # ask learner to plan new state
        allowed_action_codes = [self.actionCodes[a] for a in self.actionBasis.allowedActions(self, observedState)]
        action_code = self.learner.action_callback(state,allowed_action_codes)

        # update score
        self.score = current_score

        # update number of actions taken
        self.action_count += 1

        # save results
        if((self.save_every > 0) and (self.action_count % self.save_every == 0)):
            if self.chatter: print "Saving..."
            utils.pickle(self.learner, self.learn_file)
            
        # take action
        if self.chatter: print state, self.actions[action_code],
        action = self.actionBasis(observedState, self.actions[action_code])
        return action

# =============================================================================

class SeededGoodBadCapsuleDistanceAgent(ActionBasisAgent):
    actionBasis = basis.followActionBasis
    basis = basis.goodBadGhostCapsuleDistances
    learner_class = model_free.ModelFreeLearner
    use_initializer = True
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(SeededGoodBadCapsuleDistanceAgent, self).registerInitialState(gameState)

# =============================================================================


class GoodBadCapsuleDistanceAgent(ActionBasisAgent):
    actionBasis = basis.followActionBasis
    basis = basis.goodBadGhostCapsuleDistances
    learner_class = model_free.ModelFreeLearner
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(GoodBadCapsuleDistanceAgent, self).registerInitialState(gameState)

# =============================================================================

class SeededLocalNeighborhoodAgent(ActionBasisAgent):
    actionBasis = basis.simpleActionBasis
    basis = basis.localNeighborhood
    use_initializer = True
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(SeededLocalNeighborhoodAgent, self).registerInitialState(gameState)

# =============================================================================

class LocalNeighborhoodAgent(ActionBasisAgent):
    actionBasis = basis.simpleActionBasis
    basis = basis.localNeighborhood
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(LocalNeighborhoodAgent, self).registerInitialState(gameState)

# =============================================================================


class GhostPositionAgent(ActionBasisAgent):
    actionBasis = basis.simpleActionBasis
    basis = basis.ghostPosition
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(GhostPositionAgent, self).registerInitialState(gameState)

# =============================================================================

class HighRollerAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        "arguments given with the -a command line option will be passed here"
        pass # you probably won't need this, but just in case

    def registerInitialState(self, gameState):
        "Do any necessary initialization"
        super(HighRollerAgent, self).registerInitialState(gameState)

    def chooseAction(self, observedState):
        """
        Pacman will chase the scared ghost if present, and the nearest good
        capsule otherwise. Along the way, he will adjust his path to run into
        good ghosts.
        """

        pacmanPos = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPos,gs.getPosition())
                              for gs in ghost_states])
        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]


        # closest good capsule to Pacman
        cap = capsule.closest(observedState, self.distancer)
        print cap


        if observedState.scaredGhostPresent():
            return mapH.getDirs(pacmanPos, closest_idx)
        else:
            return mapH.getDirs(pacmanPos, cap)

class SafeAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        "arguments given with the -a command line option will be passed here"
        pass # you probably won't need this, but just in case

    def registerInitialState(self, gameState):
        "Do any necessary initialization"
        super(SafeAgent, self).registerInitialState(gameState)

    def chooseAction(self, observedState):
        """
        Pacman will eat the nearest good capsule if the ghost is not scared,
        and eat good ghosts otherwise. Along the way, he will adjust his path to
        run into good ghosts.
        """

        pacmanPos = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]

        # find the closest ghost by sorting the distances
        goodGhost = ghosts.closestGhost(observedState, self.distancer, good=False)

        # position of closest good capsule to Pacman
        closeCapsule = capsule.closest(observedState, self.distancer)

        best_action = random.choice(legalActs)
        if observedState.scaredGhostPresent():
            for dir in mapH.getDirs(pacmanPos, goodGhost):
                if dir in legalActs:
                    return dir
        else:
            for dir in mapH.getDirs(pacmanPos, closeCapsule):
                if dir in legalActs:
                    return dir
        return best_action

class BadGhostAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        "arguments given with the -a command line option will be passed here"
        pass # you probably won't need this, but just in case

    def registerInitialState(self, gameState):
        "Do any necessary initialization"
        super(BadGhostAgent, self).registerInitialState(gameState)

    def chooseAction(self, observedState):
        """
        Pacman will eat the nearest good capsule if the ghost is not scared,
        and eat good ghosts otherwise. Along the way, he will adjust his path to
        run into good ghosts.
        """

        pacmanPos = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]

        # find the closest ghost by sorting the distances
        badGhost = ghosts.getNearestBadGhost(observedState, self.distancer)

        # position of closest good capsule to Pacman
        closeCapsule = capsule.closest(observedState, self.distancer)

        best_action = random.choice(legalActs)
        if observedState.scaredGhostPresent():
            for dir in mapH.getDirs(pacmanPos, badGhost):
                if dir in legalActs:
                    return dir
        else:
            for dir in mapH.getDirs(pacmanPos, closeCapsule):
                if dir in legalActs:
                    return dir
        return best_action

class CapsuleAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        "arguments given with the -a command line option will be passed here"
        pass # you probably won't need this, but just in case

    def registerInitialState(self, gameState):
        "Do any necessary initialization"
        super(CapsuleAgent, self).registerInitialState(gameState)

    def chooseAction(self, observedState):
        "Pacman will eat the nearest good capsule."

        pacmanPos = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]

        # position of closest good capsule to Pacman
        closeCapsule = capsule.closest(observedState, self.distancer)

        best_action = random.choice(legalActs)
        for dir in mapH.getDirs(pacmanPos, closeCapsule):
            if dir in legalActs:
                return dir
        return best_action

class GhostAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        "arguments given with the -a command line option will be passed here"
        pass # you probably won't need this, but just in case

    def registerInitialState(self, gameState):
        "Do any necessary initialization"
        super(GhostAgent, self).registerInitialState(gameState)

    def chooseAction(self, observedState):
        "Pacman will eat the nearest ghost."

        pacmanPos = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]

        # position of closest good capsule to Pacman
        goodGhost = ghosts.closestGhost(observedState, self.distancer)

        best_action = random.choice(legalActs)
        for dir in mapH.getDirs(pacmanPos, goodGhost):
            if dir in legalActs:
                return dir
        return best_action

class NearestBadGhostAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        "arguments given with the -a command line option will be passed here"
        pass # you probably won't need this, but just in case

    def registerInitialState(self, gameState):
        "Do any necessary initialization"
        super(NearestBadGhostAgent, self).registerInitialState(gameState)

    def chooseAction(self, observedState):
        "Pacman will eat the nearest ghost."

        pacmanPos = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]

        # position of closest good capsule to Pacman
        badGhost = ghosts.getNearestBadGhost(observedState, self.distancer)

        best_action = random.choice(legalActs)
        for dir in mapH.getDirs(pacmanPos, badGhost):
            if dir in legalActs:
                return dir
        return best_action

# =============================================================================

# Set the exported agent here
SrcTeamAgent = ActionBasisAgent
