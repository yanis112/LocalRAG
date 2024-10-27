


class ImageAnalyzer:
    prompt = "<MORE_DETAILED_CAPTION>"

    def __init__(self, model_name="gpt-4o-mini"):
        from dotenv import load_dotenv
        import os
        import torch

        load_dotenv()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.token = os.environ["GITHUB_TOKEN"]
        self.endpoint = "https://models.inference.ai.azure.com"
        self.model_name = model_name
        self.model = None
        self.processor = None

    def load_florence_model(self):
        """Load the Florence-2 model and processor."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    def get_image_data_url(self, image_file: str, image_format: str) -> str:
        """Convert an image file to a data URL string."""
        import base64

        try:
            with open(image_file, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Could not read '{image_file}'.")
            exit()
        return f"data:image/{image_format};base64,{image_data}"

    def describe_piece(self, client, image_data_url, prompt, i, j):
        """Describe a piece of the image using OpenAI's model."""
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that describes images in details.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                "detail": "low"
                            },
                        },
                    ],
                },
            ],
            model=self.model_name,
        )
        piece_caption_text = f"({i}, {j}) description: " + response.choices[0].message.content
        print("Piece caption:", piece_caption_text)
        return piece_caption_text

    def describe_advanced(self, image_path: str, prompt: str, grid_size: int) -> str:
        """Describe an image using OpenAI's model with optional grid processing."""
        from openai import OpenAI
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        import tempfile

        client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )

        image_format = image_path.split('.')[-1]
        image_data_url = self.get_image_data_url(image_path, image_format)

        # Get global description
        global_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that describes images in details.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                "detail": "low"
                            },
                        },
                    ],
                },
            ],
            model=self.model_name,
        )
        global_caption_text = global_response.choices[0].message.content

        if grid_size > 1:
            image = Image.open(image_path)
            width, height = image.size
            piece_width, piece_height = width // grid_size, height // grid_size
            tasks = []

            with ThreadPoolExecutor() as executor:
                for i in range(grid_size):
                    for j in range(grid_size):
                        left = j * piece_width
                        upper = i * piece_height
                        right = left + piece_width
                        lower = upper + piece_height
                        piece = image.crop((left, upper, right, lower))

                        with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
                            piece.save(temp_file.name)
                            piece_data_url = self.get_image_data_url(temp_file.name, image_format)
                            tasks.append(executor.submit(self.describe_piece, client, piece_data_url, prompt, i, j))

                captions = [task.result() for task in tasks]
            return global_caption_text + "\n" + "\n".join(captions)
        else:
            return global_caption_text

    def describe(self, image_path: str, grid_size: int = 1, method: str = 'florence2') -> str:
        """Describe an image using the specified method and grid size."""
        import time
        from PIL import Image
        from tqdm import tqdm

        if method == 'florence2':
            self.load_florence_model()
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            start_time = time.time()

            inputs = self.processor(text=self.prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            global_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            global_caption = self.processor.post_process_generation(global_caption, task="<CAPTION>", image_size=(image.width, image.height))
            global_caption_text = "whole image description: " + global_caption["<CAPTION>"]

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

                    if piece.mode != 'RGB':
                        piece = piece.convert('RGB')

                    inputs = self.processor(text=self.prompt, images=piece, return_tensors="pt").to(self.device, self.torch_dtype)
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3,
                        do_sample=False
                    )
                    piece_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    piece_caption = self.processor.post_process_generation(piece_caption, task="<CAPTION>", image_size=(piece.width, piece.height))
                    piece_caption_text = f"({i}, {j}) description: " + piece_caption["<CAPTION>"]
                    print("Piece caption:", piece_caption_text)
                    captions.append(piece_caption_text)

            end_time = time.time()
            print("Time taken to generate captions: ", end_time - start_time)
            return "\n".join(captions)
        elif method in ['gpt4o', 'gpt4o-mini']:
            return self.describe_advanced(image_path, self.prompt, grid_size)
        else:
            raise ValueError("Unsupported method. Choose from 'florence2', 'gpt4o', or 'gpt4o-mini'.")

# Example usage
if __name__ == "__main__":
    import time
    analyzer = ImageAnalyzer()
    image_path = "test_ocr.png"
    prompt = """ Extract the technical information from the image. You will format it in markdown in the following way, EXEMPLE: * **BifRefNet**: A State-of-the-Art Background Removal Model  
    + [BifRefNet](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo) 🕊️ (free) is a highly performant background removal model that achieves high accuracy on various images. """

    start_time = time.time()
    description = analyzer.describe_advanced(image_path, prompt=prompt, grid_size=1)
    end_time = time.time()
    print(description)
    print("The process took:", end_time - start_time, "seconds.")