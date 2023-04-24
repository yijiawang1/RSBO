import numpy as np
import gym
import os

from gym import spaces
from gym.utils import seeding

from .gridworld_like import GridWorldBasic, SETTINGS


TREASURERANGES = {
    'item1_y': [0, 3],
    'item1_x': [7, 10],
    }


class TreasureEnv(GridWorldBasic):
    """
    Collect a valuable item to get a positive reward
    """
    #: Reward constants
    GOAL_REWARD = +10
    SUBGOAL_REWARD = +10
    PIT_REWARD = -1
    BLOCK_REWARD = 0
    STEP_REWARD = 0

    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT, PUDDLE, DOOR, KEY = list(range(9))

    def __init__(self, prob_name='it10', task={}):
        super(TreasureEnv, self).__init__(prob_name=prob_name, task=task)

        # tasks: additional items
        item1_y = np.random.randint(low=TREASURERANGES['item1_y'][0], high=TREASURERANGES['item1_y'][1])
        item1_x = np.random.randint(low=TREASURERANGES['item1_x'][0], high=TREASURERANGES['item1_x'][1])
        self._items_positions = task.get(
            'items_pos', 
            np.array([[item1_y, item1_x]])
            )
        self.num_items = self._items_positions.shape[0]
        assert self.num_items == 1, 'wrong number of treasures, must be 1'

        # state and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([self.ROWS - 1, self.COLS - 1, 1]), 
            dtype=np.float32
            )
        self.start_state = np.argwhere(self._map == self.START)[0]
        self.start_state = np.hstack((self.start_state, np.zeros((self.num_items,), dtype=np.int))) # if got the treasure
        self.state = self.start_state

        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)
        self.ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]]) #: Up, Down, Left, Right
        # extend action with extra 0's, to accommodate the dimension of state
        self.ACTIONS = np.hstack((self.ACTIONS, np.zeros((4, self.num_items), dtype=np.int)))

    def sample_tasks(self, num_tasks):
        noises = self.settings['noise_base'] + self.settings['noise_rand'] * np.random.random((num_tasks,))
        map_names, maps = [], []
        items_positions = []
        for i in range(num_tasks):
            _map_name = self.settings['map_name'] + str(np.random.randint(self.settings['map_num']))
            map_names.append(_map_name)
            maps.append(np.loadtxt(os.path.join(self.default_map_dir, _map_name + '.txt'), dtype=np.uint8))

            item1_y = np.random.randint(low=TREASURERANGES['item1_y'][0], high=TREASURERANGES['item1_y'][1])
            item1_x = np.random.randint(low=TREASURERANGES['item1_x'][0], high=TREASURERANGES['item1_x'][1])
            _items_pos = np.array([[item1_y, item1_x]])
            items_positions.append(_items_pos)
        
        tasks = [{'noise': _noise, 'map_name': _map_name, 'map': _map, 'items_pos': _items_pos}
            for (_noise, _map_name, _map, _items_pos) in zip(noises, map_names, maps, items_positions)]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._noise = task['noise']
        self._map_name = task['map_name']
        self._map = task['map']
        self._items_positions = task['items_pos']

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

        # Compute the reward and update the last self.num_items dimensions of ns
        dim = 1
        for item_pos in self._items_positions:
            dim += 1
            if (np.absolute(ns[0] - item_pos[0]) <= 0.5 and
                    np.absolute(ns[1] - item_pos[1]) <= 0.5 and
                    ns[dim] < 0.5):
                ns[dim] = 1
                r += self.SUBGOAL_REWARD
                break
        self.state = ns.copy()

        if self._map[ns[0], ns[1]] == self.GOAL:
            r += self.GOAL_REWARD
        if self._map[ns[0], ns[1]] == self.PIT:
            r += self.PIT_REWARD

        done = self.isDone()
        return ns, r, done, {'task': self._task}




        
