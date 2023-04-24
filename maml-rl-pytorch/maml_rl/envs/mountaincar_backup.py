import numpy as np
import gym

from gym import spaces
from gym.utils import seeding


SETTINGS = {
    'noise_base': 0, 'noise_rand': 0.01, 
    'episodeCap': 200,
    'discount_factor': 1.,
    'start_low': -0.6, 'start_high': -0.4,
}

class MountainCarEnv(gym.Env):
    """
    The goal is to drive an under accelerated car up to the hill.\n
    **STATE:**        Position and velocity of the car [x, xdot] \n
    **ACTIONS:**      [Acc backwards, Coast, Acc forward] \n
    **TRANSITIONS:**  Move along the hill with some noise on the movement. \n
    **REWARD:**       -1 per step and 0 at or beyond goal (``x-goal > 0``). \n
    There is optional noise on vehicle acceleration.
    **REFERENCE:**
    Based on `RL-Community Java Implementation <http://library.rl-community.org/wiki/Mountain_Car_(Java)>`_
    """

    state_space_dims = 2
    continuous_dims = [0, 1]

    XMIN = -1.2  # : Lower bound on domain position
    XMAX = 0.6  #: Upper bound on domain position
    XDOTMIN = -0.07  # : Lower bound on car velocity
    XDOTMAX = 0.07  #: Upper bound on car velocity
    INIT_STATE = np.array([-0.5, 0.0])  # : Initial car state
    STEP_REWARD = -1  # : Penalty for each step taken before reaching the goal
    GOAL_REWARD = 0  #: Reward for reach the goal.
    #: X-Position of the goal location (Should be at/near hill peak)
    GOAL = .5
    
    accelerationFactor = 0.001  # : Magnitude of acceleration action
    gravityFactor = -0.0025
    #: Hill peaks are generated as sinusoid; this is freq. of that sinusoid.
    hillPeakFrequency = 3.0

    # Used for visual stuff:
    domain_fig = None
    valueFunction_fig = None
    policy_fig = None
    actionArrow = None
    X_discretization = 20
    XDot_discretization = 20
    CAR_HEIGHT = .2
    CAR_WIDTH = .1
    ARROW_LENGTH = .2

    def __init__(self, task={}):
        super(MountainCarEnv, self).__init__()
        
        self.settings = SETTINGS
        self.episodeCap = self.settings['episodeCap']
        self.discount_factor = self.settings['discount_factor']
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
        self.observation_space = spaces.Box(
            low=np.array([self.XMIN, self.XDOTMIN]), 
            high=np.array([self.XMAX, self.XDOTMAX]), 
            dtype=np.float32
            )
        self.INIT_STATE[0] = self._start
        self.state = self.INIT_STATE

        self.num_actions = 3
        self.action_space = spaces.Discrete(self.num_actions)
        self.ACTIONS = [-1, 0, 1]
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        return self.state

    def step(self, a):
        assert self.action_space.contains(a)
        
        position, velocity = self.state
        noise = self.accelerationFactor * self._noise * 2 * (np.random.rand() - .5)
        velocity += (noise +
                     self.ACTIONS[a] * self.accelerationFactor +
                     np.cos(self.hillPeakFrequency * position) * self.gravityFactor)
        velocity = bound(velocity, self.XDOTMIN, self.XDOTMAX)
        position += velocity
        position = bound(position, self.XMIN, self.XMAX)
        if position <= self.XMIN and velocity < 0:
            velocity = 0  # Bump into wall
        ns = np.array([position, velocity])
        self.state = ns.copy()
        done = self.isDone()
        r = self.GOAL_REWARD if done else self.STEP_REWARD

        done = self.isDone()
        return ns, r, done, {'task': self._task}

    def isDone(self, s=None):
        if self.steps >= self.settings['episodeCap']:
            return True
        return self.state[0] > self.GOAL

    def possibleActions(self, s=None):
        return np.arange(self.num_actions)


def bound(x, m, M=None):
    """
    :param x: scalar

    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)