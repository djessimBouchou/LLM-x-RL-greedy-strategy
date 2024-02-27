### TO CONTINUE ###

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

translation_generation_config = GenerationConfig(
    num_beams=4,
    early_stopping=True,
    decoder_start_token_id=0,
    eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,
)

translation_generation_config.save_pretrained("/tmp", "translation_generation_config.json")
generation_config = GenerationConfig.from_pretrained("/tmp", "translation_generation_config.json")

def LLM(prompt, max_tokens=256, stop=None, temperature=0.0, device = "cuda"):

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**encoded, max_new_tokens=max_tokens, generation_config=generation_config)
    output= tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return output

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



init_episodes = 100
max_episodes = 200
temperature = 0.0
max_context = 1020  # In tokens.

# Memory bank with reward-labeled episodes: each is a list of state-action tuples.
episodes = []
rewards = []

env = CartPoleEnv()

# Generate some random policy rollouts and add them to memory.
while len(episodes) < init_episodes:
  episode = []
  s = env.reset()
  while not env.terminated:
    a = env.random_act()
    episode.append((s, a))
    s = env.step(a)
  episodes.append(episode)
  rewards.append(env.reward)

# Incremental rollouts with the LLM in the loop.
while len(episodes) < max_episodes:

  # Set a desired reward for the current rollout.
  desired_reward = np.max(rewards) + 20 + np.int32(np.random.uniform() * 10)
  prompt = f"{desired_reward}:"

  # Environment reset.
  state = env.reset()
  buffer = []

  while not env.terminated and env.reward < 200:
    prompt += f"{env.state_to_str(state)},"
    num_tokens = len(tokenizer.encode(prompt))

    # Build context of episodes sorted by ascending rewards.
    context = ""
    for i in np.argsort(rewards)[::-1]:
      if num_tokens + 10 > max_context:  # Each episode should have at least 10 tokens.
        break
      episode, reward = episodes[i], rewards[i]
      size = min(len(episode), (max_context - num_tokens) // 5)
      text = f"{reward}:" + ",".join([f"{env.state_to_str(s)},{env.act_to_str(a)}" for s, a in episode[:size]])
      num_tokens += 2 + size * 5   # Manual math here to count tokens. Calling the tokenizer too much can get slow.
      context = f"{text}\n{context}"

    # LLM inference.
    pred = LLM(context + prompt, max_tokens=4, stop=[",", "\n"], temperature=temperature)

    # If predicted action is invalid, sample random action.
    # Alternatively, one can sample the LLM from only the set of valid actions
    # by using a logit bias. This is the approach we take in our experiments.
    try:
      act = env.str_to_act(pred.strip(","))
    except:
      act = -1
    if act not in [0, 1]:
      print(f"Invalid action '{pred}'. Sampling random one.")
      act = env.random_act()

    prompt += f"{env.act_to_str(act)},"
    buffer.append((state, act))

    # Show LLM input.
    print(context + prompt)
    print("---------------------------------------------------------")
    print("Num episodes:", len(episodes), "Curr highest return:", np.max(rewards))
    print("---------------------------------------------------------")

    # Step environment.
    state = env.step(act)

  episodes.append(buffer)
  rewards.append(env.reward)

  # Make a plot of performance over time.
  plt.scatter(np.arange(init_episodes), rewards[:init_episodes], c="gray", alpha=0.3)
  plt.scatter(np.arange(init_episodes, len(rewards)), rewards[init_episodes:], alpha=0.3)
  max_over_time = [rewards[init_episodes]]
  for reward in rewards[init_episodes+1:]:
    max_over_time.append(max(reward, max_over_time[-1]))
  plt.plot(np.arange(init_episodes, len(rewards)), max_over_time)
  plt.axhline(y=200, color='gray', linestyle='--', alpha=0.3)
  plt.show()