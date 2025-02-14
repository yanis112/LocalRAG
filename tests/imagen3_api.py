
import os
#import google.generativeai as genai
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Only run this block for Gemini Developer API
client = genai.Client(api_key=os.getenv('IMAGEN_API_KEY'))

response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()

