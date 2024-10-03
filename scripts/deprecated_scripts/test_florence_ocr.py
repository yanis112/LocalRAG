import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Check if GPU is available and move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load image
path="test_factures/test_screen.png"

image = Image.open(path).convert("RGB")

#coupe l'image en deux 2 images distinctes
image1 = image.crop((0, 0, image.width, image.height//2))
image2 = image.crop((0, image.height//2, image.width, image.height))



def run_example(task_prompt,image,text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Process inputs and move them to GPU
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate text
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,
        num_beams=3
    )

    # Decode and post-process the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    print(parsed_answer)


if __name__ == "__main__":
    
    
    #prompt = "<OCR>"
    prompt="<OCR_WITH_REGION>"
    run_example(task_prompt=prompt,image=image1)
    run_example(task_prompt=prompt,image=image2)
  
    