# ----------------------------------------------------------------
#               LLM + DQN on MiniGrid Empty Environment
# ----------------------------------------------------------------

from minigrid.envs import EmptyEnv
from minigrid.wrappers import FullyObsWrapper

from captionner_minigrid import CaptionnerGT
from transformers import AutoModelForCausalLM, AutoTokenizer


def description_game(captionner, obs, num_actions):

    description = "You play an agent on a 2-dimensional rectangular grid world of size {}x{}, observed with a top-down point of view.\n".format(obs["image"].shape[0], obs["image"].shape[1])
    description += "The four corners are : North-East (1, 1), North-West (1, {}), South-East ({}, 1) and South-West ({}, {}).".format(obs["image"].shape[0], obs["image"].shape[1], obs["image"].shape[0], obs["image"].shape[1])
    description += "Here is a description of what is on the grid.\n\n"
    caption = captionner.caption(obs)
    possible_actions = "The possible actions are : Turn 90 deg left, Turn 90 deg right and Move forward.\n\n"
    final_query = "Produce a reasoning to understand the different steps to reach the final goal. \n"
    final_query += "Finally produce a clear list of {} next actions to take from the described scenario, starting with ACTIONS TO TAKE : + insert list of actions.".format(num_actions)

    return description + caption + possible_actions + final_query

captionner = CaptionnerGT()

env = EmptyEnv(size = 20)
env = FullyObsWrapper(env)

obs, _ = env.reset()
prompt = description_game(captionner, obs, num_actions=20)

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": prompt},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])