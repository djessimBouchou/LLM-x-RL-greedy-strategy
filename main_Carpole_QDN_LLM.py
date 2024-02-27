# ----------------------------------------------------------------
#                  LLM + DQN  on Carpole
# ----------------------------------------------------------------


from transformers import AutoModelForCausalLM, AutoTokenizer

from carpole_env import CartPoleEnv

from models.DQN import DQNAgent


# 1. LLM Model used to generate some actions

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# 2. Create environment

env = CartPoleEnv()


# 3. Create DQN Agent

dqn_carpole = DQNAgent(network_type="MlpPolicy", env= env, llm = model, tokenizer=tokenizer)
dqn_carpole.train(num_frames=10000)