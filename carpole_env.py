import gymnasium as gym
import numpy as np

class CartPoleEnv:

  def __init__(self):
    self.env = gym.make("CartPole-v1", max_episode_steps=200)  #, render_mode="human"

    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space
    
    self.reset()

  def reset(self, seed = None):
    obs, info = self.env.reset(seed = seed)
    self.terminated = False
    self.reward = 0
    self.state = self.norm_state(obs)  # Pole angle and velocity.
    return obs, info

  def step(self, act):
    obs, reward, terminated, truncated, info = self.env.step(act)
    self.state = self.norm_state(obs)
    self.terminated = terminated or truncated
    self.reward += np.int32(reward)
    return obs, reward, terminated, truncated, info

  def random_act(self):
    return self.env.action_space.sample()

  def norm_state(self, state):
    p = (np.clip(state[2], -0.25, 0.25) + 0.25) * (100 / 0.5)
    v = (np.clip(state[3], -3, 3) + 3) * (100 / 6)
    return int(np.round(p)), int(np.round(v))

  def state_to_str(self, state):
    return f" {state[0]} {state[1]}"

  def act_to_str(self, act):
    # LLMs bias on 0 so make the actions 1 and 2 instead.
    return f" {act + 1}"

  def str_to_act(self, str):
    return int(str) - 1

