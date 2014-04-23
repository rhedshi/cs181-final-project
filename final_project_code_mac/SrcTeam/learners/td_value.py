# Implements model-free learning (Q-learning)

import numpy as np
import numpy.random as npr
import sys
import math
import random

class TDValueLearner:

    def __init__(self, basis_dimensions, actions):

        self.basis_dimensions = basis_dimensions
        self.actions = actions

        # default values for hyperparameters
        self.alpha = 0.1
        self.gamma = 0.1
        self.epsilon = 0

        # state of MDP
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # dimensions of s
        dims = self.basis_dimensions

        # learned value of state s
        self.V = np.zeros(dims)

        # learned reward of state s
        self.R = np.zeros(dims + (len(self.actions),))

        # empirical distribution for estimating transition model
        # self.N[s + a] = number of times we've taken action a from state s
        self.N = np.ones(dims + (len(self.actions),))

        # self.Np[s + a + sp] = number of times we've transitioned to state sp
        # after taking action a in state s
        self.Np = np.zeros(dims + (len(self.actions),) + dims)

        # note that to calculate the empirical distribution of the transition model P(sp | s,a),
        # you can do:
        #     self.Np[ s + a + (Ellipsis,) ] / self.N[(Ellipsis,) + a]

        'Number of times taken action a from each state s'
        self.k = np.ones(dims + (len(self.actions),))

    def reset(self):
        # reset state of MDP
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state, actions):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''


        # store state, last state for learning in reward_callback
        self.last_state  = self.current_state
        self.current_state = state
        s  = state

        # plan
        if (random.random() < self.epsilon):
            # with some probability self.epsilon, just pick a random action
            new_action = random.choice(actions)
        else:
            # otherwise plan based on the learned transition model
            # array of expected values for each possible action
            expected_values = np.array([ np.dot( (self.Np[ s + (a,) + (Ellipsis,) ] / self.N[(Ellipsis,) + (a,)]).flat, self.V.flat ) for a in self.actions ])

            # pick the new action pi(s) as the action with the largest expected value
            er = self.R[s + (Ellipsis,)] + expected_values
            new_action_index =  np.argmax(er[np.array(actions)])
            new_action = actions[new_action_index]
        assert new_action in actions

        # learn the transition model
        if (self.last_state != None):
            s  = self.last_state
            sp = self.current_state
            a  = (self.last_action,)

            self.N[s + a] += 1
            self.Np[s + a + sp] += 1

        # store last action, record exploration
        self.last_action = new_action
        a  = (self.last_action,)
        self.k[s + a] += 1

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        if (self.last_state != None) and (self.current_state != None) and (self.last_action != None):
            s  = self.last_state
            sp = self.current_state
            a  = (self.last_action,)

            # lower alpha over time as we visit more frequently
            # alpha = 1.0 / self.k[s + a]
            alpha = 0.01

            # update V
            self.V[s] = self.V[s] + alpha * ( (reward + self.gamma * self.V[sp]) - self.V[s] )

            # update R with a "running average"
            self.R[s + a] = (self.R[s + a] * (self.k[s + a] - 1) + reward) / (self.k[s + a])


        self.last_reward = reward


    def bin(self, value, range, bins):
        '''Divides the interval between range[0] and range[1] into equal sized
        bins, then determines in which of the bins value belongs'''
        bin_size = (range[1] - range[0]) / bins
        return math.floor((value - range[0]) / bin_size)
