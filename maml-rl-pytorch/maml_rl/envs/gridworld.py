import numpy as np
import gym
import os

from gym import spaces
from gym.utils import seeding

from .gridworld_like import GridWorldBasic, SETTINGS


class GridWorldEnv(GridWorldBasic):

    #: Reward constants
    GOAL_REWARD = +1
    PIT_REWARD = -1
    BLOCK_REWARD = 0
    STEP_REWARD = 0

    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT = list(range(6))

    def __init__(self, prob_name='gw10Two1', task={}):
        super(GridWorldEnv, self).__init__(prob_name=prob_name, task=task)

        # state and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([self.ROWS - 1, self.COLS - 1]), 
            dtype=np.float32
            )
        self.start_state = np.argwhere(self._map == self.START)[0]
        self.state = self.start_state

        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)
        self.ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]]) #: Up, Down, Left, Right

    def step(self, a):
        assert self.action_space.contains(a)
        if np.random.rand() < self._noise: # Random Move
            a = np.random.choice(self.possibleActions())
        ns = self.state + self.ACTIONS[a] # Take action
        self.steps += 1
        r = self.STEP_REWARD

        # Check bounds on state values
        if (ns[0] < 0 or ns[0] == self.ROWS or
                ns[1] < 0 or ns[1] == self.COLS or
                self._map[ns[0], ns[1]] == self.BLOCKED):
            ns = self.state.copy()
        else: # If in bounds, update the current state
            self.state = ns.copy()

        # Compute the reward
        if self._map[ns[0], ns[1]] == self.GOAL:
            r = self.GOAL_REWARD
        if self._map[ns[0], ns[1]] == self.PIT:
            r = self.PIT_REWARD

        done = self.isDone()
        return ns, r, done, {'task': self._task}
