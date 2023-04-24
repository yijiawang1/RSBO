import numpy as np
import gym
import os

from gym import spaces
from gym.utils import seeding


SETTINGS = {
    'gw10Two1': {
        'map_name': '10x10_TwoRooms1_', 'map_num': 5,
        'noise_base': 0, 'noise_rand': 0.02,
        'max_steps': 1000,
        'episodeCap': 500,
        'discount_factor': 1.,
        },
    'gw20Three1': {
        'map_name': '20x20_ThreeRooms1_', 'map_num': 18,
        'noise_base': 0.02, 'noise_rand': 0.02,
        'max_steps': 10000,
        'episodeCap': 1000,
        'discount_factor': 1.,
        },
    'it10': {
        'map_name': '10x10_SmallRoom1_', 'map_num': 1,
        'noise_base': 0, 'noise_rand': 0.0,
        'max_steps': 2000,
        'episodeCap': 500,
        'discount_factor': 0.98,
        },
    'ky10One': {
        'map_name': '10x10_TwoRooms_ii', 'map_num': 2,
        'noise_base': 0, 'noise_rand': 0.0,
        'max_steps': 1000,
        'episodeCap': 500,
        'discount_factor': 0.999,
        },
}


class GridWorldBasic(gym.Env):
    """
    The GridWorld domain simulates a path-planning problem for a mobile robot
    in an environment with obstacles. The goal of the agent is to
    navigate from the starting point to the goal state.
    The map is loaded from a text file filled with numbers showing the map with the following
    coding for each cell:

    * 0: empty
    * 1: blocked
    * 2: start
    * 3: goal
    * 4: pit

    **STATE:**
    The Row and Column corresponding to the agent's location. \n
    **ACTIONS:**
    Four cardinal directions: up, down, left, right (given that
    the destination is not blocked or out of range). \n
    **TRANSITION:**
    There is 30% probability of failure for each move, in which case the action
    is replaced with a random action at each timestep. Otherwise the move succeeds
    and the agent moves in the intended direction. \n
    **REWARD:**
    The reward on each step is -.001 , except for actions
    that bring the agent to the goal with reward of +1.\n
    """
    
    _map = start_state = goal = None
    
    #: Number of rows and columns of the map
    ROWS = COLS = 0

    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT = list(range(6))

    # directory of maps shipped with rlpy
    default_map_dir = os.path.join(os.getcwd(), "../problems/rlpy/Domains/GridWorldMaps")

    def __init__(self, prob_name, task={}):
        super(GridWorldBasic, self).__init__()

        self.prob_name = prob_name
        self.settings = SETTINGS[prob_name]
        self.seed()

        # tasks
        self._task = task
        self._noise = task.get(
            'noise', 
            self.settings['noise_base'] + self.settings['noise_rand'] * np.random.random()
            )
        self._map_name = task.get(
            'map_name',
            self.settings['map_name'] + str(np.random.randint(self.settings['map_num']))
            )
        self._map = task.get(
            'map', 
            np.loadtxt(os.path.join(self.default_map_dir, self._map_name + '.txt'), dtype=np.uint8)
            )
        if self._map.ndim == 1: self._map = self._map[np.newaxis, :]

        self.ROWS, self.COLS = np.shape(self._map)
        self.steps = 0

        # state and action spaces
        self.observation_space = None
        self.start_state = None
        self.state = None

        self.num_actions = None
        self.action_space = None
        self.ACTIONS = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        noises = self.settings['noise_base'] + self.settings['noise_rand'] * np.random.random((num_tasks,))
        map_names, maps = [], []
        for i in range(num_tasks):
            _map_name = self.settings['map_name'] + str(np.random.randint(self.settings['map_num']))
            map_names.append(_map_name)
            maps.append(np.loadtxt(os.path.join(self.default_map_dir, _map_name + '.txt'), dtype=np.uint8))
        
        tasks = [{'noise': _noise, 'map_name': _map_name, 'map': _map}
            for (_noise, _map_name, _map) in zip(noises, map_names, maps)]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._noise = task['noise']
        self._map_name = task['map_name']
        self._map = task['map']

    def reset(self):
        self.state = self.start_state
        self.steps = 0
        return self.state

    def step(self, a):
        pass

    def isDone(self, s=None):
        if s is None:
            s = self.state
        if self._map[int(s[0]), int(s[1])] == self.GOAL: return True
        if self._map[int(s[0]), int(s[1])] == self.PIT: return True
        if self.prob_name is None: # 'gw20Three1'
            if self.steps >= self.settings['max_steps']: return True
        else:
            if self.steps >= self.settings['episodeCap']: return True
        
        return False

    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        possibleA = np.array([], np.uint8)
        for a in range(self.num_actions):
            ns = s + self.ACTIONS[a]
            if (ns[0] < 0 or ns[0] == self.ROWS or
                ns[1] < 0 or ns[1] == self.COLS or
                self._map[int(ns[0]), int(ns[1])] == self.BLOCKED):
                continue
            possibleA = np.append(possibleA, [a])
        return possibleA
