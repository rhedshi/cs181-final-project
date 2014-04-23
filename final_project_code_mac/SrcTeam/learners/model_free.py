 # Implements model-free learning (Q-learning)

import numpy as np
import numpy.random as npr
import sys
import math
import random

class ModelFreeLearner:

    def __init__(self, basis_dimensions, actions):

        # basis function
        self.basis_dimensions = basis_dimensions
        self.actions = actions

        # hyperparameters
        self.alpha = 0.001
        self.gamma = 0.1
        self.epsilon = 0.1

        # state parameters
        self.current_action = None
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # epoch number
        self.iter = 0

        # dimensionality
        dims = self.basis_dimensions + (len(actions),)
        self.Q = np.zeros(dims)

        # Number of times taken action a from each state s (for adaptive 
        # learning rate)
        self.k = np.ones(dims)

    def reset(self):
        self.current_action = None
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.iter += 1

    def action_callback(self, state, actions):
        '''Implement this function to learn things and take actions.
        Return an element of actions'''

        # epsilon-greedy policy
        if (random.random() < self.epsilon):
            new_action = random.choice(actions)
        else:
            new_action_index = np.argmax(self.Q[state][np.array(actions)])
            new_action = actions[new_action_index]
        assert new_action in actions
        new_state  = state

        # store action and state transition for learning
        self.last_action = new_action
        self.last_state  = self.current_state
        self.current_state = new_state

        s  = state
        a  = (self.last_action,)
        self.k[s + a] += 1

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        if (self.last_state != None) and (self.current_state != None) and (self.last_action != None):
            s  = self.last_state
            sp = self.current_state
            a  = (self.last_action,)

            if self.iter < 100:
                alpha = self.alpha
            else:
                alpha = self.alpha*0.1

            # learn Q
            self.Q[s + a] = self.Q[s + a] + alpha * (reward + self.gamma * np.max(self.Q[sp]) - self.Q[s + a] )

        self.last_reward = reward
