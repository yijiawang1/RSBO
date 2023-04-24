import math
import numpy as np
import gym

from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import MountainCarEnv as MountainCarEnv_


SETTINGS = {
    'noise_base': 0, 'noise_rand': 0.01,
    'max_steps': 10000,
    'episodeCap': 200,
    'discount_factor': 1.,
    'start_low': -0.6, 'start_high': -0.4,
}

class MountainCarEnv(MountainCarEnv_):

    INIT_STATE = np.array([-0.5, 0.0])

    def __init__(self, task={}):
        super(MountainCarEnv, self).__init__()
        
        self.settings = SETTINGS
        self.seed()

        # tasks
        self._task = task
        self._noise = task.get(
            'noise', 
            self.settings['noise_base'] + self.settings['noise_rand'] * np.random.random()
            )
        self._start = task.get(
            'start',
            self.settings['start_low'] + (self.settings['start_high'] - self.settings['start_low']) * np.random.rand()
            )

        self.steps = 0

        # state and action spaces
        self.INIT_STATE[0] = self._start
        self.state = self.INIT_STATE
        

    def sample_tasks(self, num_tasks):
        noises = self.settings['noise_base'] + self.settings['noise_rand'] * np.random.random((num_tasks,))
        # print((self.settings['start_high'] - self.settings['start_low']))
        starts = (
            self.settings['start_low'] + 
            (self.settings['start_high'] - self.settings['start_low']) * np.random.random((num_tasks,))
            )
        
        tasks = [{'noise': _noise, 'start': _start}
            for (_noise, _start) in zip(noises, starts)]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._noise = task['noise']
        self._start = task['start']

    def reset(self):
        self.state = self.INIT_STATE
        self.steps = 0
        return np.array(self.state)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        noise = self.force * self._noise * 2 * (np.random.rand() - .5)
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = -1.0

        self.state = (position, velocity)
        self.steps += 1
        # if self.steps >= self.settings['episodeCap']: done = True
        if self.steps >= self.settings['max_steps']: done = True
        return np.array(self.state), reward, done, {'task': self._task}
