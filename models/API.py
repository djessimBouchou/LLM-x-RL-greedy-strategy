
import base64
import json
from openai import OpenAI
from PIL import Image
import io
import google.generativeai as genai
import numpy as np

class GeminiSession():
    def __init__(self, key_path, dict_action, descript_game,  model_name='gemini-pro') -> None:
        """
            key_path: path to the json file containing the api key (with the api_key field)
        """


        with open(key_path, 'r') as file:
            api_key = json.load(file)["api_key"]

        genai.configure(api_key=api_key)

        self.description_game = descript_game
        self.action_to_value = dict_action

        # self.action_to_value = {
        #     "up": 0,
        #     "right": 1,
        #     "down": 2,
        #     "left": 3
        # }
        
    def call_vision(self, list_prompt):
        """
        Call Gemini Vision Pro. list_prompt should be a list or string or numpy array 3D 
        """
        self.model = genai.GenerativeModel("gemini-pro-vision")
        for i in range(len(list_prompt)):
            if type(list_prompt[i]) == str:
                continue
            elif type(list_prompt[i]) == np.ndarray:
                list_prompt[i] = Image.fromarray(list_prompt[i])
            else:
                raise Exception("Only str or numpy array plzzz")
            
            response = self.model.generate_content(list_prompt)
            response.resolve()

        return response.text
    
    def call(self, prompt):
        self.model = genai.GenerativeModel("gemini-pro")
        response = self.model.generate_content(prompt)
        response.resolve()
        return response.text
    

    def generate_list_of_actions(self, image, nb_actions = 10):

        list_possible_actions = list(self.action_to_value.keys())
        prompt = self.description_game

        message = "\n The possible actions are the following : " + ", ".join(list_possible_actions) + "."
        message += "\n Can you give me {} actions to realize, in order to get closer to the goal?".format(nb_actions)

        prompt += message

        response = self.send(image, prompt)

        list_actions = response.split()
        list_actions = [action.strip(', ') for action in list_actions if action.strip(', ') in list_possible_actions]
        list_actions = [self.action_to_value[action] for action in list_actions]

        return list_actions
    

class GPTSession():
    def __init__(self, key_path) -> None:
        with open(key_path, 'r') as file:
            api_key = json.load(file)["api_key"]
        
        self.client = OpenAI(api_key=api_key)

        self.action_to_value = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3
        }


    def call(self, prompt, model_name = "gpt-4-turbo-preview"):
        resp = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        return resp.choices[0].message.content


    def generate_list_of_actions(self, description, nb_actions = 10):

        list_possible_actions = list(self.action_to_value.keys())
        prompt = description

        message = "\n The possible actions are the following : " + ", ".join(list_possible_actions) + "."
        message += "\n Can you give me {} actions to realize, in order to get closer to the goal?".format(nb_actions)

        prompt += message

        response = self.call(prompt)

        print(response)

        list_actions = response.split()
        list_actions = [action.strip(',') for action in list_actions if action.strip(', ') in list_possible_actions]
        list_actions = [self.action_to_value[action] for action in list_actions]

        return list_actions
    


if __name__ == "__main__":
    api = GPTSession(key_path = "key.json")
    
    print("OK")