
import base64
import json
from openai import OpenAI
from PIL import Image
import io
import google.generativeai as genai

class API_Gemini():
    def __init__(self, key_path, dict_action, descript_game,  model_name='gemini-pro-vision') -> None:
        """
            key_path: path to the json file containing the api key (with the api_key field)
        """


        with open(key_path, 'r') as file:
            api_key = json.load(file)["api_key"]

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        self.description_game = descript_game
        self.action_to_value = dict_action

        # self.action_to_value = {
        #     "up": 0,
        #     "right": 1,
        #     "down": 2,
        #     "left": 3
        # }
        
    def send(self, image, prompt):
        image = Image.fromarray(image)
        response = self.model.generate_content([prompt, image])
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
