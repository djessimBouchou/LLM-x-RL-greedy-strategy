
import base64
import json
from openai import OpenAI
from PIL import Image
import io
import google.generativeai as genai

class API_Gemini():
    def __init__(self, key_path, model_name='gemini-pro-vision') -> None:
        """
            key_path: path to the json file containing the api key (with the api_key field)
        """


        with open(key_path, 'r') as file:
            api_key = json.load(file)["api_key"]

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        self.action_to_value = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3
        }
        
    def send(self, image, prompt):
        image = Image.fromarray(image)
        response = self.model.generate_content([prompt, image])
        response.resolve()

        return response.text