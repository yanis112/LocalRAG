from PIL import Image
from groq import Groq
import google.generativeai as genai
import os
from typing import Optional, List, Union
from concurrent.futures import ThreadPoolExecutor
import tempfile
import time
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

class ImageAnalyzerAgent:
    """
    A class for analyzing and describing images using various AI models.

    This class provides a unified interface to interact with different vision models,
    including Groq, Hugging Face's Florence-2, and Google's Gemini.
    It supports optional grid processing for more detailed image descriptions.

    Attributes:
        prompt (str): The default prompt used for image descriptions.
        device (str): The device to run models on, either 'cuda:0' or 'cpu'.
        torch_dtype (torch.dtype): The data type for torch tensors.
        token (str): API token for accessing the OpenAI service.
        endpoint (str): Endpoint URL for the OpenAI service.
        model_name (str): The name of the model to use for description if not precised in the method call.
        model (transformers.PreTrainedModel): The loaded model if using a florence2 based model.
        groq_token (str): API key for accessing the Groq service.
        processor (transformers.PreTrainedProcessor): The processor for preparing inputs for the model.
        groq_client (Groq): The client for accessing the Groq service.
        gemini_token (str): The API key for accessing the Google Gemini service.

    Methods:
        __init__(model_name="gpt-4o-mini"): Initializes the ImageAnalyzerAgent.
        load_florence_model(): Loads the Florence-2 model and processor.
        get_image_data_url(image_file: str, image_format: str) -> str: Converts an image file to a data URL string.
        _describe_piece(client, image_data_url, prompt, i, j) -> str: Describes a piece of the image using OpenAI's model.
        resize_image(image: Image, max_size: int = 512) -> Image: Resizes the image while maintaining aspect ratio.
        _describe_openai(image_path: str, prompt: str, grid_size: int, model_name:str="gpt-4o-mini", max_size: int = 768) -> str: Describes an image using OpenAI's model with optional grid processing.
        _describe_with_groq(image_path: str, prompt: str, model_name: str="llama-3.2-90b-vision-preview", max_size: int = 768) -> str: Describes an image using Groq's vision model.
        _describe_gemini(image_path: str, prompt: str, model_name: str="gemini-1.5-flash") -> str: Describes an image using a specified Gemini model.
        _describe_with_florence(image_path: str, prompt: str, grid_size: int) -> str: Describes an image using the florence2 model.
        describe(image_path: str, prompt: Optional[str] = None, vllm_provider: str = "florence2", model_name: Optional[str]=None , grid_size: int = 1) -> str: Unified method to describe an image using a specified provider.
    """

    prompt = "<MORE_DETAILED_CAPTION>"

    def __init__(self):
        """Initializes the ImageAnalyzerAgent."""
        from dotenv import load_dotenv

        load_dotenv()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.token = os.environ["GITHUB_TOKEN"]
        self.endpoint = "https://models.inference.ai.azure.com"
        self.groq_token = os.environ.get("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_token) if self.groq_token else None
        self.gemini_token = os.environ.get("GOOGLE_API_KEY")
        if self.gemini_token:
            genai.configure(api_key=self.gemini_token)

    def load_florence_model(self):
         """Load the Florence-2 model and processor."""
         self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)
         self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )
    
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
    
    def _describe_piece(self, client, image_data_url, prompt, i, j) -> str:
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

    def resize_image(self, image: Image, max_size: int = 512) -> Image:
        """Resize image if it exceeds maximum dimension while maintaining aspect ratio."""
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image
    
    def _describe_openai(self, image_path: str, prompt: str, grid_size: int, model_name:str="gpt-4o-mini", max_size: int = 768) -> str:
        """
        Describe an image using OpenAI's model with optional grid processing.
    
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            grid_size (int): The number of grid cells in a row/column (for grid processing).
            model_name (str): The name of the OpenAI model to use.
            max_size (int): Maximum size of the image dimension before resizing.
    
        Returns:
            str: The generated image description text.
        """
        from openai import OpenAI
        
        client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
        
        # Open and resize the image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.resize_image(image, max_size)
        
        # Save resized image to temporary file
        image_format = image_path.split('.')[-1]
        with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
            image.save(temp_file.name)
            image_data_url = self.get_image_data_url(temp_file.name, image_format)
        
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
            model=model_name,
        )
        global_caption_text = global_response.choices[0].message.content
        
        if grid_size > 1:
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
                            tasks.append(executor.submit(self._describe_piece, client, piece_data_url, prompt, i, j))
                
                captions = [task.result() for task in tasks]
            return global_caption_text + "\n" + "\n".join(captions)
        else:
            return global_caption_text

    def _describe_with_groq(self, image_path: str, prompt: str, model_name: str="llama-3.2-90b-vision-preview", max_size: int = 768) -> str:
        """
        Describe an image using Groq's vision model.
    
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            model_name (str): The name of the Groq model to use.
            max_size (int): Maximum size of the image dimension before resizing.
    
        Returns:
            str: The generated image description text.
        """
        
        # Open and resize image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.resize_image(image, max_size)
        
        # Save and get data URL
        image_format = image_path.split('.')[-1]
        with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
            image.save(temp_file.name)
            image_data_url = self.get_image_data_url(temp_file.name, image_format)
    
        # Get global description
        completion = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=1024
        )
        
        return completion.choices[0].message.content
    
    def _describe_gemini(self, image_path: str, prompt: str, model_name: str="gemini-1.5-flash") -> str:
         """
         Describes an image using a specified Gemini model.
    
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            model_name (str): The name of the Gemini model to use.
        Returns:
             str: The generated image description text.
         """
         try:
             # Load the image
             image = Image.open(image_path)
             if image.mode != 'RGB':
                 image = image.convert('RGB')
             model = genai.GenerativeModel(model_name)
             response = model.generate_content([prompt, image])
             return response.text
         except Exception as e:
             return f"Error during Gemini description: {e}"
    
    def _describe_with_florence(self, image_path: str, prompt: str, grid_size: int) -> str:
        """Describe an image using the florence2 model."""
        self.load_florence_model()
        image = Image.open(image_path)
        if image.mode != 'RGB':
                image = image.convert('RGB')
        start_time = time.time()
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
             input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
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
                inputs = self.processor(text=prompt, images=piece, return_tensors="pt").to(self.device, self.torch_dtype)
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
                piece_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                piece_caption = self.processor.post_process_generation(piece_caption, task="<CAPTION>", image_size=(piece.width, piece.height))
                piece_caption_text = f"({i}, {j}) description: " + piece_caption["<CAPTION>"]
                print("Piece caption:", piece_caption_text)
                captions.append(piece_caption_text)
        end_time = time.time()
        print("Time taken to generate captions: ", end_time - start_time)
        return "\n".join(captions)
    
    def describe(self, image_path: str, prompt: Optional[str] = None, vllm_provider: str = "florence2", vllm_name: Optional[str]=None, grid_size: int = 1) -> str:
        """
        Unified method to describe an image using a specified provider.

        Args:
            image_path (str): Path to the image file.
            prompt (str, optional): The prompt to guide the image description. Defaults to None.
            vllm_provider (str, optional): The provider of the vision model. Defaults to "florence2".
            model_name (str, optional): The name of the model to use. Defaults to None.
            grid_size (int, optional): The number of grid cells in a row/column for grid processing. Defaults to 1.

        Returns:
            str: The generated image description text.

        Raises:
            ValueError: If an unsupported `vllm_provider` is specified.
        """
        if prompt is None:
            prompt = self.prompt

        if vllm_provider == "florence2":
            return self._describe_with_florence(image_path, prompt, grid_size)
        elif vllm_provider == "groq":
            if not self.groq_client:
                raise ValueError("Groq API key not found in environment variables")
            return self._describe_with_groq(image_path, prompt, model_name= vllm_name if vllm_name else "llama-3.2-90b-vision-preview")
        elif vllm_provider == "github":
             return self._describe_openai(image_path, prompt, grid_size, model_name=vllm_name if vllm_name else 'gpt-4o-mini')
        elif vllm_provider == "gemini":
             return self._describe_gemini(image_path, prompt, model_name= vllm_name if vllm_name else "gemini-1.5-flash")
        else:
            raise ValueError(
                "Unsupported provider. Choose from 'florence2', 'groq', 'github' or 'gemini'."
            )
# Example usage
if __name__ == "__main__":
    analyzer = ImageAnalyzerAgent()
    image_path = "aux_data/document_scores.png"
    prompt = "Describe precisely the content of the image."

    # Example using Groq
    start_time = time.time()
    description_groq = analyzer.describe(
        image_path, prompt=prompt, vllm_provider="groq", vllm_name="llama-3.2-90b-vision-preview"
    )
    end_time = time.time()
    print(f"Groq Description:\n{description_groq}")
    print("Groq process took:", end_time - start_time, "seconds.")
    
    print("#############################################")

    # Example using Gemini
    start_time = time.time()
    description_gemini = analyzer.describe(
        image_path, prompt=prompt, vllm_provider="gemini", vllm_name="gemini-1.5-flash"
    )
    end_time = time.time()
    print(f"\nGemini Description:\n{description_gemini}")
    print("Gemini process took:", end_time - start_time, "seconds.")
    
    print("#############################################")

    # Example using github
    start_time = time.time()
    description_github = analyzer.describe(
        image_path, prompt=prompt, vllm_provider="github", vllm_name="gpt-4o-mini"
    )
    end_time = time.time()
    print(f"\nGithub Description:\n{description_github}")
    print("Github process took:", end_time - start_time, "seconds.")

   