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
        self.learn_file = (kwargs['file'] if 'file' in kwargs else 'SrcTeam/data/learn_'+self.__class__.__name__)
        self.save_every = (kwargs['save_every'] if 'save_every' in kwargs else 100)

    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(SrcTeamAgent, self).registerInitialState(gameState)

        # get all the allowed actions, encode them for learners
        self.actions = gameState.getLegalPacmanActions()
        self.actionCodes = { action: i for (i, action) in enumerate(self.actions) }

        # ---------------------------------------------------------------------
        # pick a basis function here
        self.basis = basis.ghostDistance
        # ---------------------------------------------------------------------

        # remember basis function dimensions
        self.basis_dimensions = self.basis.dimensions

        # try to load the learner from a file
        if(self.learn_file and os.path.isfile(self.learn_file)):
            self.learner = utils.unpickle(self.learn_file)
            self.learner.reset()
        else:
            # ---------------------------------------------------------------------
            # pick a learner here
            self.learner = model_free.ModelFreeLearner(self.basis_dimensions, self.actionCodes.values())
            # self.learner = td_value.TDValueLearner(self.basis_dimensions, self.actionCodes.values())
            # ---------------------------------------------------------------------

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

        # update number of actions taken
        self.action_count += 1

        # save results
        if((self.save_every > 0) and (self.action_count % self.save_every == 0)):
            utils.pickle(self.learner, self.learn_file)

        # take action
        return self.actions[action_code]

# =============================================================================

class ActionBasisAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """

    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        self.learn_file = (kwargs['file'] if 'file' in kwargs else 'SrcTeam/data/learn_'+self.__class__.__name__)
        self.save_every = (kwargs['save_every'] if 'save_every' in kwargs else 100)

    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        super(ActionBasisAgent, self).registerInitialState(gameState)

        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")

        # ---------------------------------------------------------------------
        # pick an action basis function here
        # self.actionBasis = basis.followActionBasis
        self.actionBasis = basis.simpleActionBasis
        # ---------------------------------------------------------------------

        # get all the allowed actions, encode them for learners
        self.actions = self.actionBasis.allActions[:]
        self.actionCodes = { action: i for (i, action) in enumerate(self.actions) }

        # ---------------------------------------------------------------------
        # pick a basis function here
        self.basis = basis.ghostDistance
        # ---------------------------------------------------------------------

        # remember basis function dimensions
        self.basis_dimensions = self.basis.dimensions

        # try to load the learner from a file
        if(self.learn_file and os.path.isfile(self.learn_file)):
            self.learner = utils.unpickle(self.learn_file)
            self.learner.reset()
        else:
            # ---------------------------------------------------------------------
            # pick a learner here
            self.learner = model_free.ModelFreeLearner(self.basis_dimensions, self.actionCodes.values())
            # self.learner = td_value.TDValueLearner(self.basis_dimensions, self.actionCodes.values())
            # ---------------------------------------------------------------------

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
        reward = last_score - current_score

        # pass reward to learner
        self.learner.reward_callback(reward)

        # apply basis function to calculate new state
        state = self.basis(self,observedState)

        # ask learner to plan new state
        allowed_action_codes = [self.actionCodes[a] for a in self.actionBasis.allowedActions(self, observedState)]
        action_code = self.learner.action_callback(state,allowed_action_codes)

        # update score
        self.last_score = current_score

        # update number of actions taken
        self.action_count += 1

        # save results
        if((self.save_every > 0) and (self.action_count % self.save_every == 0)):
            utils.pickle(self.learner, self.learn_file)
            
        # take action
        return self.actionBasis(self, observedState, self.actions[action_code])

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
        capsule = capsule.closest()
        print capsule


        if scaredGhostPresent():
            return mapH.getDirs(pacmanPos, closest_idx)
        else:
            return mapH.getDirs(pacmanPos, capsule)

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
