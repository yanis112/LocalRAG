from json import load
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/enhanceaiteam/Flux-Uncensored-V2"
#"https://api-inference.huggingface.co/models/XLabs-AI/flux-RealismLora"
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
headers = {"Authorization": "Bearer " + TOKEN}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def generate_image(prompt, height=1024, width=1024, guidance_scale=10.0, num_inference_steps=20, **parameters):
    payload = {
        "inputs": prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        **parameters
    }
    image_bytes = query(payload)
    return Image.open(io.BytesIO(image_bytes))

def save_image(image, file_path):
    image.save(file_path)

# Example usage
if __name__ == "__main__":
    prompt = """High quality selfie portrait of a japanese student with large tits, short skirt, beautiful face, long black hair, background is a classic student room in a mess, clothes scattered on the floor, papers on the bed, we see her nude boobs as she pull up her t-shirt"""
    image = generate_image(prompt)
    print("Image size:", image.size)
    save_image(image, "flux-dev.png")