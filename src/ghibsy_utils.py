import requests
from dotenv import load_dotenv
import os
load_dotenv()

token=os.getenv('HUGGINGFACE_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/aleksa-codes/flux-ghibsky-illustration"
headers = {"Authorization": "Bearer {}".format(token)}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

image_bytes = query({
	"inputs": """GHIBSKY style, Epic duel between two man: clash on a volcanic island. One man, wearing a white coat over a crimson suit, wields swirling lava that scorches the earth. His opponent, clad in a white coat and ice-blue suit, summons glacial powers that freeze the air. Vibrant orange magma contrasts with shimmering blue ice, their powers colliding in a spectacular display. The scene is painted with soft, dreamy colors and intricate details characteristic of Miyazaki's art. Swirling steam rises where fire meets frost, creating a misty, ethereal atmosphere. In the background, a turbulent ocean and ash-filled sky frame the intense battle"""
})
print("Image bytes:", image_bytes)
# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
#plot the image
#save the image in the temp folder

image.save("image.jpg")
from matplotlib import pyplot as plt
plt.imshow(image)
plt.axis('off')

