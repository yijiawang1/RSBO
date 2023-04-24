import numpy as np
import gym
import os

from gym import spaces
from gym.utils import seeding

from .gridworld_like import GridWorldBasic, SETTINGS


class KeyDoorEnv(GridWorldBasic):
    """
    The GridWorld_Key domain simulates a path-planning problem for a mobile robot
    in an environment with obstacles. The goal of the agent is to
    navigate from the starting point to the goal state.
    """
    #: Reward constants
    GOAL_REWARD = +1
    KEY_REWARD = 0
    PIT_REWARD = -1
    BLOCK_REWARD = 0
    STEP_REWARD = 0

    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT, PUDDLE, DOOR, KEY = list(range(9))

    def __init__(self, prob_name='ky10One', task={}):
        super(KeyDoorEnv, self).__init__(prob_name=prob_name, task=task)

        # state and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([self.ROWS - 1, self.COLS - 1, 1]), 
            dtype=np.float32
            )
        self.start_state = np.argwhere(self._map == self.START)[0]
        self.start_state = np.hstack((self.start_state, np.zeros((1, ), dtype=np.int))) # if got the key
        self.state = self.start_state

        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)
        self.ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]]) #: Up, Down, Left, Right
        # extend action with an extra 0, to accommodate the dimension of state
        self.ACTIONS = np.hstack((self.ACTIONS, np.zeros((4, 1), dtype=np.int))) 

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
            r += self.BLOCK_REWARD
        elif (ns[2] < 0.5 and self._map[ns[0], ns[1]] == self.DOOR): # no key
            ns = self.state.copy()
            r += self.BLOCK_REWARD

        # Compute the reward and update the state
        if ns[2] < 0.5 and self._map[ns[0], ns[1]] == self.KEY:
            ns[2] = 1
            # r += self.KEY_REWARD
        self.state = ns.copy()

        if self._map[ns[0], ns[1]] == self.GOAL:
            r += self.GOAL_REWARD
        if self._map[ns[0], ns[1]] == self.PIT:
            r += self.PIT_REWARD

        done = self.isDone()
        return ns, r, done, {'task': self._task}
