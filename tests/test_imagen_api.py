from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Création du client pour le Gemini Developer API
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"),
                      http_options={'api_version': 'v1alpha'})

# Appel de génération d'images avec la configuration correcte
response = client.models.generate_images(
    model="imagen-3.0-generate-002",
    prompt="Fuzzy bunnies in my kitchen",
    config=types.GenerateImagesConfig(
        number_of_images=4,
        output_mime_type="image/jpeg"
    )
)

print("Response: ", response)

# Affichage des images générées
for generated_image in response.generated_images:
    image = Image.open(BytesIO(generated_image.image.image_bytes))
    image.show()
