
import base64
import json
from openai import OpenAI
from PIL import Image
import io

class API_Session():
    def __init__(self, key_path, model_name="gpt-4-vision-preview") -> None:
        """
            key_path: path to the json file containing the api key (with the api_key field)
        """


        with open(key_path, 'r') as file:
            api_key = json.load(file)["api_key"]

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

        self.action_to_value = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3
        }
        
    def send(self, image, prompt):


        image = Image.fromarray(image)
        image_stream = io.BytesIO()
        image.save(image_stream, format='PNG')
        image_binary = image_stream.getvalue()
        image_base64 = base64.b64encode(image_binary).decode('utf-8')

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Give me exactly 10 random actions from [up, right, down, left]. The format of your answer should be very strict, we accept only this format of answer: action1 - action2 -..."},
                    # {
                    # "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{image_base64}",
                    #     },
                    # },
                    ],
                }
            ],
            max_tokens=300,
            )
        
        message = response.choices[0].message.content
        message_split = message.split(" - ")

        return list(map(lambda x: self.action_to_value[x], message_split))