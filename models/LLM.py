from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import torch
import re

class LLM():

    def __init__(self, name_model = "mistralai/Mistral-7B-Instruct-v0.2", device = "cuda", max_context = 2000, max_tokens = 4):
        
        self.max_context = max_context
        self.max_tokens = max_tokens
        self.device = device

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(name_model, quantization_config=quantization_config, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)

        self.episodes = []
        self.rewards = []
        self.current_episode = []
        self.prompt = []

    def reset(self):
        self.current_episode = []
        self.generate_desired_reward()

    def select_action(self, env, state, nb_actions_selected = 10):

        num_tokens = len(self.tokenizer.encode(self.prompt))

        # Build context of episodes sorted by ascending rewards.
        context = ""
        for i in np.argsort(self.rewards)[::-1]:
            if num_tokens + 10 > self.max_context:  # Each episode should have at least 10 tokens.
                break
            episode, reward = self.episodes[i], self.rewards[i]
            size = min(len(episode), (self.max_context - num_tokens) // 5)
            text = f"{reward}:" + ",".join([f"{env.state_to_str(s)},{env.act_to_str(a)}" for s, a in episode[:size]])
            num_tokens += 2 + size * 5   # Manual math here to count tokens. Calling the tokenizer too much can get slow.
            context = f"{text}\n{context}"

        input_LLM = context + self.prompt + f"{env.state_to_str(env.norm_state(state))},"

        # Pass request through the LLM
        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer(input_LLM, return_tensors="pt").to(self.device)
            output = self.model.generate(**encoded, max_length=len(encoded[0]) + self.max_tokens, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
            print(output.shape)
            output = output[0, len(encoded[0]):]
            print(output.shape)
            output= self.tokenizer.decode(output, skip_special_tokens=True)
            print("Output : " + output)

        # Convert to list of actions
        pred = re.sub(r"[, ]", "", output)
        try:
            act = env.str_to_act(pred.strip()[0])
        except:
            act = -1
        if act not in [0, 1]:
            print(f"Invalid action '{pred}'. Sampling random one.")
            act = env.random_act()
        LLM_next_selected_actions = [act]

        print(context + self.prompt)
        print("---------------------------------------------------------")
        print("Num episodes:", len(self.episodes), "Curr highest return:", np.max(self.rewards))
        print("---------------------------------------------------------")


        # output = output.split(',')
        # output = [e.strip() for e in output]
        
        # print("Output LLM generation : ", output)
        
        # LLM_next_selected_actions = []
        # num_actions = 0
        # i = 0
        # while num_actions < nb_actions_selected and i < len(output):
        #     if output[i] == "1" or output[i] == "2":
        #         LLM_next_selected_actions.append(env.str_to_act(output[i]))
        #         num_actions += 1
        #     i += 1

        # print("List of actions outputed by LLM : ", LLM_next_selected_actions)

        return LLM_next_selected_actions
    
    def generate_desired_reward(self):
        if len(self.rewards) > 0:
            desired_reward = np.max(self.rewards) + 20 + np.int32(np.random.uniform() * 10)
        else :
            desired_reward = 50 + np.int32(np.random.uniform() * 10)
        self.prompt = f"{desired_reward}:"

    def add_episode_done(self, score):
        self.rewards.append(np.int32(score))
        self.episodes.append(self.current_episode)
        self.reset()

    def update(self, env, s, a):
        self.prompt += f"{env.state_to_str(env.norm_state(s))},"
        self.prompt += f"{env.act_to_str(a)},"
        self.current_episode.append((env.norm_state(s), a))


