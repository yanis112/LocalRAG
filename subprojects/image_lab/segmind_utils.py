import requests
from PIL import Image
from io import BytesIO
import os

class FluxModel:
    def __init__(self, url="https://api.segmind.com/v1/flux-1.1-pro"):
        self.api_key = "SG_30578a2e5b366a30" # "SG_7701bec37468f66b"  #"SG_8c5843047188180b"
        self.url = url
        self.headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
        }

    def generate(self, prompt, seed=123, width=1024, height=1024, aspect_ratio="1:1", output_format="png", output_quality=80, safety_tolerance=5, prompt_upsampling=False):
        data = {
            'seed': seed,
            'width': width,
            'height': height,
            'prompt': prompt,
            'aspect_ratio': aspect_ratio,
            'output_format': output_format,
            'output_quality': output_quality,
            'safety_tolerance': safety_tolerance,
            'prompt_upsampling': prompt_upsampling
        }
        response = requests.post(self.url, json=data, headers=self.headers)
        print("API response:", response)
        image = Image.open(BytesIO(response.content))
        return image

    def save(self, image, filename="generated_image.png"):
        temp_folder = os.path.join(os.path.expanduser("~"), "temp")
        os.makedirs(temp_folder, exist_ok=True)
        file_path = os.path.join(temp_folder, filename)
        image.save(file_path, format="PNG")
        print(f"Image saved to {file_path}")

if __name__ == "__main__":
    flux_model = FluxModel()
    image = flux_model.generate(prompt="A high quality blockbuster movie scene featuring a tolkien silmarilion scene with a dwarf king clad in magnificient armour fighting a huge Balrog, 7 meters tall, clad in darkness and flames, dark skin, flamming whip and sword, forwrad curves horns. Scene takes place in carved in the moutain stone kingdom..")
    flux_model.save(image)
