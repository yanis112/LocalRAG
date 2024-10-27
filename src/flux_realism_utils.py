import requests
from dotenv import load_dotenv
import os

token=os.getenv('HUGGINGFACE_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/XLabs-AI/flux-RealismLora"
headers = {"Authorization": "Bearer {}".format(token)}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": """A breathtaking image depicting a tree trunk covered in lush moss and mushrooms within a vibrant, green forest. The scene is bathed in soft, diffused light, highlighting the rich textures and vivid shades of green. Striking details can be seen in the mushrooms, with delicately sculpted caps and gills. Overlaid is a subtle network motif resembling neural connections or circuitry, seamlessly blending nature with technology. This expert composition in 8K quality makes for an ideal presentation background, uniting the serene beauty of untouched nature with a modern tech-inspired pattern."""
     #"""A breathtaking image depicting a tree trunk covered in lush moss and mushrooms within a vibrant, green forest. The scene is bathed in soft, diffused light, highlighting the rich textures and vivid shades of green. The mushrooms feature striking detail, with delicately sculpted caps and gills. Expert composition and impressive realism make this an ideal 8K quality photograph for presentation backgrounds, capturing the serene and mystical beauty of untouched nature"""
})
# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))

#save the image in the temp folder
image.save("flux_realism.jpg")