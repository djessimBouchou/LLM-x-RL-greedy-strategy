import gymnasium as gym
import numpy as np

class CartPoleEnv:

  def __init__(self):
    self.env = gym.make("CartPole-v1")  #, render_mode="human"
    self.reset()

  def reset(self):
    obs, info = self.env.reset()
    self.terminated = False
    self.reward = 0
    self.state = self.norm_state([obs[2], obs[3]])  # Pole angle and velocity.
    return self.state

  def step(self, act):
    obs, reward, terminated, truncated, info = self.env.step(act)
    self.state = self.norm_state([obs[2], obs[3]])
    self.terminated = terminated or truncated
    self.reward += np.int32(reward)
    return self.state

  def random_act(self):
    return self.env.action_space.sample()

  def norm_state(self, state):
    p = (np.clip(state[0], -0.25, 0.25) + 0.25) * (100 / 0.5)
    v = (np.clip(state[1], -3, 3) + 3) * (100 / 6)
    return int(np.round(p)), int(np.round(v))

  def state_to_str(self, state):
    return f" {state[0]} {state[1]}"

  def act_to_str(self, act):
    # LLMs bias on 0 so make the actions 1 and 2 instead.
    return f" {act + 1}"

  def str_to_act(self, str):
    return int(str) - 1

