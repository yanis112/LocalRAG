import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load and quantize the model
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def describe(image, grid_size=1):
    prompt = "<MORE_DETAILED_CAPTION>"
    
    start_time = time.time()
    
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Generate global caption
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    global_caption = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    global_caption = processor.post_process_generation(global_caption, task="<CAPTION>", image_size=(image.width, image.height))
    global_caption_text = "whole image description: " + global_caption["<CAPTION>"]
    
    # Split image into grid_size x grid_size pieces
    width, height = image.size
    piece_width, piece_height = width // grid_size, height // grid_size
    captions = [global_caption_text]
    
    for i in tqdm(range(grid_size), desc="Processing rows"):
        for j in range(grid_size):
            left = j * piece_width
            upper = i * piece_height
            right = left + piece_width
            lower = upper + piece_height
            piece = image.crop((left, upper, right, lower))
            
            # Ensure each piece is in RGB format
            if piece.mode != 'RGB':
                piece = piece.convert('RGB')
            
            # Generate caption for each piece
            inputs = processor(text=prompt, images=piece, return_tensors="pt").to(device, torch_dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            piece_caption = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            piece_caption = processor.post_process_generation(piece_caption, task="<CAPTION>", image_size=(piece.width, piece.height))
            piece_caption_text = f"({i}, {j}) description: " + piece_caption["<CAPTION>"]
            print("Piece caption:", piece_caption_text)
            captions.append(piece_caption_text)
    
    end_time = time.time()
    print("Time taken to generate captions: ", end_time - start_time)
    return "\n".join(captions)

# Example usage
if __name__ == "__main__":
    
    #open test.png
    image = Image.open("test.png")
    
    print(describe(image, grid_size=4))