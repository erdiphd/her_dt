import gym
from gym import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, agent_location=None, target_location=None, dimensions=2, reward_type="dense"):
        self.dimensions = dimensions
        self.reward_type = reward_type

        if dimensions == 2 and (len(agent_location) == 3 or len(target_location) == 3):
            raise ValueError("dimensions == 2 but list size == 3")
        if dimensions == 3 and (len(agent_location) == 2 or len(target_location) == 2):
            raise ValueError("dimensions == 3 but list size == 2")

        # default locations
        # if dimensions == 2:
        #     if target_location is None:
        #         target_location = [3, 3]
        #     if agent_location is None:
        #         agent_location = [3, 0]
        #
        # if dimensions == 3:
        #     if target_location is None:
        #         target_location = [3, 3, 0]
        #     if agent_location is None:
        #         agent_location = [3, 0, 0]

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

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        if self.dimensions == 2:
            return {
                "distance": ((self._agent_location[0] - self._target_location[0])**2
                             + (self._agent_location[1] - self._target_location[1])**2)**0.5
            }
        if self.dimensions == 3:
            return {
                "distance": ((self._agent_location[0] - self._target_location[0])**2
                             + (self._agent_location[1] - self._target_location[1])**2
                             + (self._agent_location[2] - self._target_location[2])**2)**0.5
            }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed) # TODO: why does it work for ge_q_ts but not here?
        super().seed(seed)
        # reset locations to default
        self._agent_location = self.start
        self._target_location = self.goal

        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # self._agent_location = np.random.randint(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = np.random.randint(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        #  “right”, “up”, “left”, “down”.
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        # Sparse rewards
        reward = 0
        if self.reward_type == "sparse":
            if terminated:
                reward = 1
            else:
                reward = 0

        # Dense rewards
        elif self.reward_type == "dense":
            if self.dimensions == 2:
                reward = (-1) * (((self._agent_location[0] - self._target_location[0]) ** 2
                                  + (self._agent_location[1] - self._target_location[1]) ** 2) ** 0.5)

            if self.dimensions == 3:

                reward = (-1) * (((self._agent_location[0] - self._target_location[0]) ** 2
                                  + (self._agent_location[1] - self._target_location[1]) ** 2
                                  + (self._agent_location[2] - self._target_location[2]) ** 2) ** 0.5)

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, False, info
