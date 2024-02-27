# ----------------------------------------------------------------
#               LLM + DQN on MiniGrid Empty Environment
# ----------------------------------------------------------------

from minigrid.envs import EmptyEnv
from minigrid.wrappers import FullyObsWrapper

from captionner_minigrid import CaptionnerGT
from transformers import AutoModelForCausalLM, AutoTokenizer


def description_game(captionner, obs, num_actions):

    description = "You play an agent on a 2-dimensional grid world of size {}x{}.\nHere is a description of what is on the grid.\n\n".format(obs["image"].shape[0], obs["image"].shape[1])
    caption = captionner.caption(obs)
    final_query = "Produce a reasoning to understand the different steps to reach the final goal. \n"
    final_query += "After that, output clearly the {} next actions to take. An action can be (go up, go down, go right, go left)".format(num_actions)

    return description + caption + final_query

captionner = CaptionnerGT()

env = EmptyEnv(size = 20)
env = FullyObsWrapper(env)

obs, _ = env.reset()
prompt = description_game(captionner, obs)

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