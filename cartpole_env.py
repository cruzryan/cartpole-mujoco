import time
import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class CartPoleEnv:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path('./yeah.xml')
        self.d = mujoco.MjData(self.m)
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        self.done = False
        self.max_steps = 1000
        self.current_step = 0
    
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.done = False
        self.current_step = 0
        return self.get_observation()

    def get_observation(self):
        pole_angle = self.d.qpos[1]  # hinge joint angle
        cart_position = self.d.qpos[0]
        return np.array([cart_position, pole_angle], dtype=np.float32)

    def step(self, action):
        # Apply action
        self.d.ctrl[0] = action  # Assuming action is a continuous value for force

        mujoco.mj_step(self.m, self.d)

        self.current_step += 1
        observation = self.get_observation()
        reward = 1.0 - (abs(observation[1] - np.pi / 2))  # Reward is high for pole near 90 degrees

        if self.current_step >= self.max_steps or abs(observation[0]) > 1.0:  # End if max steps or cart out of bounds
            self.done = True

        return observation, reward, self.done, {}

    def render(self):
        with self.viewer.lock():
            pass
        self.viewer.sync()

    def close(self):
        self.viewer.close()

# Usage:
# env = CartPoleEnv()
# state = env.reset()
# action = 0  # Example action
# next_state, reward, done, _ = env.step(action)
