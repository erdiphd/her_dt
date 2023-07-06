import math

import gym
from gym import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size, agent_location, target_location, dimensions, reward_type, obstacle_is_on, env):
        self.dimensions = dimensions
        self.reward_type = reward_type
        """
        Obstacle positions:
        FetchPush: 1.25 0.7 0.44 -> size: 0.1 0.025 0.04
        FetchSlide: 1.1 0.8 0.44 -> size: 0.025 0.2 0.05
        FetchPickAndPlace: 1.20 0.7 0.44 -> size: 0.05 0.025 0.04
        """
        self.env = env
        if env == "push":
            # FetchPush
            self.obstacle_cell_2 = [13, 7]
            self.obstacle_cell_1 = [12, 7]
        elif env == "pick":
            # FetchPickAndPlace
            self.obstacle_cell_1 = [12, 7]
        elif env == "slide":
            # FetchSlide
            self.obstacle_cell_1 = [11, 7]
            self.obstacle_cell_2 = [11, 8]
            self.obstacle_cell_3 = [11, 9]
        self.obstacle_is_on = obstacle_is_on

        if obstacle_is_on is True and dimensions == 3:
            raise ValueError('decision tree cannot work with 3 dimensions and an obstacle')

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        # initialize start and goal
        self.start = agent_location
        self.goal = target_location
        self._agent_location = agent_location
        self._target_location = target_location

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        if dimensions == 2:
            self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                }
            )

            # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
            self.action_space = spaces.Discrete(4)

            """
            The following dictionary maps abstract actions from `self.action_space` to 
            the direction we will walk in if that action is taken.
            I.e. 0 corresponds to "right", 1 to "up" etc.
            """
            self._action_to_direction = {
                0: np.array([1, 0]),
                1: np.array([0, 1]),
                2: np.array([-1, 0]),
                3: np.array([0, -1]),
            }

        if dimensions == 3:
            self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, size - 1, shape=(3,), dtype=int),
                    "target": spaces.Box(0, size - 1, shape=(3,), dtype=int),
                }
            )
            # We have 6 actions, corresponding to "right", "forward", "left", "backward", "right", "up", "down"
            self.action_space = spaces.Discrete(6)

            """
            The following dictionary maps abstract actions from `self.action_space` to 
            the direction we will walk in if that action is taken.
            I.e. 0 corresponds to "right", 1 to "up" etc.
            """
            self._action_to_direction = {
                0: np.array([1, 0, 0]),
                1: np.array([0, 1, 0]),
                2: np.array([-1, 0, 0]),
                3: np.array([0, -1, 0]),
                4: np.array([0, 0, 1]),
                5: np.array([0, 0, -1]),
            }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        if self.dimensions == 2:
            return {
                "distance": ((self._agent_location[0] - self._target_location[0]) ** 2
                             + (self._agent_location[1] - self._target_location[1]) ** 2) ** 0.5
            }
        if self.dimensions == 3:
            return {
                "distance": ((self._agent_location[0] - self._target_location[0]) ** 2
                             + (self._agent_location[1] - self._target_location[1]) ** 2
                             + (self._agent_location[2] - self._target_location[2]) ** 2) ** 0.5
            }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().seed(seed)

        # reset locations to default
        self._agent_location = self.start
        self._target_location = self.goal

        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # self._agent_location = np.random.randint(0, self.size, size=3, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = np.random.randint(
        #         0, self.size, size=3, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        #  “right”, “up”, “left”, “down”.

        direction = self._action_to_direction[action]
        old_agent_location = self._agent_location.copy()
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        collision_occurred = False
        # check if obstacle test is turned on
        if self.obstacle_is_on is True:
            # check if agent_position and obstacle position intersect
            if self.env == "slide":
                if np.array_equal(self._agent_location, self.obstacle_cell_1) or np.array_equal(self._agent_location, self.obstacle_cell_2) or np.array_equal(self._agent_location, self.obstacle_cell_3):
                    collision_occurred = True
            if self.env == "push":
                if np.array_equal(self._agent_location, self.obstacle_cell_1) or np.array_equal(self._agent_location, self.obstacle_cell_2):
                    collision_occurred = True
            if self.env == "pick":
                if np.array_equal(self._agent_location, self.obstacle_cell_1):
                    collision_occurred = True
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(np.array(self._agent_location), np.array(self._target_location))
        # Sparse rewards
        reward = 0
        if self.reward_type == "sparse":
            if collision_occurred is False:
                if terminated:
                    reward = 1
                else:
                    reward = 0
            else:
                reward = -1

        # Dense rewards
        elif self.reward_type == "dense":
            if self.obstacle_is_on is True:
                if collision_occurred is False:
                    # dimensions can only be 2. There are no obstacle tests with 3D.
                    # FetchPick uses 2D with obstacle for x-y coordinate and 3D without obstacle for z-coordinate.
                    if self.dimensions == 2:
                        if (((self._agent_location[0] - self._target_location[0]) ** 2
                                            + (self._agent_location[1] - self._target_location[1]) ** 2) ** 0.5) == 0:
                            reward = 1
                        else:
                            reward = (-0.1) / (((self._agent_location[0] - self._target_location[0]) ** 2
                                            + (self._agent_location[1] - self._target_location[1]) ** 2) ** 0.5)
                else:
                    reward = -1
            else:
                if self.dimensions == 2:
                    reward = (-1) * (((self._agent_location[0] - self._target_location[0]) ** 2
                                      + (self._agent_location[1] - self._target_location[1]) ** 2) ** 0.5)
                if self.dimensions == 3:
                    reward = (-1) * (((self._agent_location[0] - self._target_location[0]) ** 2
                                      + (self._agent_location[1] - self._target_location[1]) ** 2
                                      + (self._agent_location[2] - self._target_location[2]) ** 2) ** 0.5)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
