import gymnasium as gym
import numpy as np

class DubinsEnv(gym.Env):
    """
    A simple implementation of the Dubins car problem.
    The agent can move forward, turn left, or turn right.
    The goal is to reach the target without colliding with obstacles.
    """
    def __init__(self):
        super(DubinsEnv, self).__init__()
        # speed of the car
        self.speed = 0.5

        # State bounds.
        self.left_boundary = 0.0
        self.right_boundary = 5.0
        self.bounds = np.array([[0.0, 5.0], [0.0, 3.0], [0, 2 * np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # defining failure set
        self.lane_bottom = 0.0
        self.lane_top = 3.0
        self.obstacle1_center = np.array([2.0, 1.0])
        self.obstacle1_radius = 0.5
        self.obstacle2_center = np.array([4.0, 2.0])
        self.obstacle2_radius = 0.5

        # defining target set
        self.target_x = 5.0

        # self.bounds = np.array([[-1.1, 1.1], [-1.1, 1.1], [0, 2 * np.pi]])
        # self.low = self.bounds[:, 0]
        # self.high = self.bounds[:, 1]
        self.sample_inside_obs = False
        self.sample_inside_tar = True

        self.n_zero_order_hold = 10 # number of steps to hold the zero action before querying LM again

        # Gym variables.
        self.action_space = gym.spaces.Discrete(3)
        midpoint = (self.low + self.high) / 2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(
            np.float32(midpoint - interval/2),
            np.float32(midpoint + interval/2),
        )
        self.midpoint = midpoint
        self.interval = interval