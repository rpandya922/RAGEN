import gymnasium as gym
import numpy as np
import re

def calculate_margin_circle(s, c_r, negativeInside=True):
    """Calculates the margin to a circle in the x-y state space.

        Args:
            s (np.ndarray): the state of the agent. It requires that s[0] is the
                x position and s[1] is the y position.
            c_r (tuple of np.ndarray and float)): (center, radius).
            negativeInside (bool, optional): add a negative sign to the distance
                if inside the circle. Defaults to True.

        Returns:
            float: margin.
        """
    center, radius = c_r
    dist_to_center = np.linalg.norm(s[:2] - center)
    margin = dist_to_center - radius

    if negativeInside:
        return margin
    else:
        return -margin

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

        # Dubins car parameters.
        self.time_step = 0.05
        self.R_turn = 0.6

        self.max_turning_rate = self.speed / self.R_turn  # w
        self.discrete_controls = np.array([
            -self.max_turning_rate, 0., self.max_turning_rate
        ])

        self.doneType = "toEnd"

        self.INVALID_ACTION = -1

    # == Compute Margin ==
    def safety_margin(self, s):
        """Computes the margin (e.g. distance) between the state and the failue set.

        Args:
            s (np.ndarray): the state of the agent.

        Returns:
            float: negative numbers indicate being inside the failure set (safety
                violation).
        """
        # s = [x, y, theta]
        left_margin = s[0] - self.left_boundary
        bottom_margin = s[1] - self.lane_bottom
        top_margin = self.lane_top - s[1]

        obstacle1_margin = calculate_margin_circle(
            s, [self.obstacle1_center, self.obstacle1_radius], negativeInside=True
        )
        obstacle2_margin = calculate_margin_circle(
            s, [self.obstacle2_center, self.obstacle2_radius], negativeInside=True
        )
        safety_margin = min(
            left_margin, bottom_margin, top_margin, obstacle1_margin, obstacle2_margin
        )

        return safety_margin

    def target_margin(self, s):
        """Computes the margin (e.g. distance) between the state and the target set.

        Args:
            s (np.ndarray): the state of the agent.

        Returns:
            float: positive numbers indicate reaching the target. If the target set
                is not specified, return None.
        """
        return s[0] - self.target_x

    def reset(self):
        theta_rnd = np.random.uniform(0, 2 * np.pi)
        x_rnd = np.random.uniform(self.left_boundary, self.right_boundary)
        y_rnd = np.random.uniform(self.lane_bottom, self.lane_top)

        self.state = np.array([x_rnd, y_rnd, theta_rnd])

        return self.state
    
    def integrate_forward(self, state, u):
        x, y, theta = state

        x = x + self.time_step * self.speed * np.cos(theta)
        y = y + self.time_step * self.speed * np.sin(theta)
        theta = np.mod(theta + self.time_step * u, 2 * np.pi)

        state = np.array([x, y, theta])
        return state

    def step(self, action):
        u = self.discrete_controls[action]
        state_nxt = self.integrate_forward(self.state, u)
        self.state = state_nxt
        l_x = self.target_margin(self.state[:2])
        g_x = self.safety_margin(self.state[:2])

        fail = g_x < 0
        success = l_x >= 0

        # = `done` signal
        if self.doneType == "toEnd":
            done = not self.car.check_within_bounds(self.state)
        elif self.doneType == "fail":
            done = fail
        elif self.doneType == "TF":
            done = fail or success
        else:
            raise ValueError("invalid done type!")

        # = `info`
        if done and self.doneType == "fail":
            info = {"g_x": self.penalty * self.scaling, "l_x": l_x}
        else:
            info = {"g_x": g_x, "l_x": l_x}
        
        # NOTE: for now, the reward is set as min(l_x, g_x) so it can be used for non-safety RL 

        return np.copy(self.state), min(l_x, g_x), done, info
    
    def extract_action(self, text):
        """
        Extract action from text.
        - -1: Still (Invalid Action)
        - 0: Right
        - 1: Straight
        - 2: Left
        """
        DIRECTION_MAP = {"Right": 0, "Straight": 1, "Left": 2}
        # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
        # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
        pattern = r'^\s*(([0-2])\s*\((right|straight|left)\)|(right|straight|left)|([0-2]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return self.INVALID_ACTION
        
        if match.group(2):
            return int(match.group(2))
        elif match.group(4): 
            return DIRECTION_MAP[match.group(4).capitalize()]
        elif match.group(5): 
            return int(match.group(5))
        
        return self.INVALID_ACTION