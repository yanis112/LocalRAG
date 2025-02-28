from PIL import Image
from groq import Groq
import google.generativeai as genai
import os
from typing import Optional
import tempfile
import time
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
)
from azure.core.credentials import AzureKeyCredential

class ImageAnalyzerAgent:
    """
    A class for analyzing and describing images using various AI models.

    This class provides a unified interface to interact with different vision models,
    including Groq, Hugging Face's Florence-2, Google's Gemini.

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
    """

    prompt = "<MORE_DETAILED_CAPTION>"

    def __init__(self, config=None):
        """
        Initializes the ImageAnalyzerAgent.
        
        Args:
            config (dict, optional): Configuration dictionary containing generation parameters.
                                   Defaults to None.
        """
        from dotenv import load_dotenv
        
        self.config = config

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
        
        # Default refinement prompt template path
        self.default_refinement_prompt_path = "prompts/image_refinement_prompt.txt"

    def load_florence_model(self):
         """Load the Florence-2 model and processor."""
         self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)
         self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )
    
    def load_refinement_prompt(self, prompt_path: str) -> PromptTemplate:
        """Load and return the refinement prompt template."""
        with open(prompt_path, "r", encoding='utf-8') as f:
            template = f.read()
        return PromptTemplate.from_template(template)

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
    
    def _describe_openai(self, image_path: str, prompt: str, model_name:str="gpt-4o-mini", max_size: int = 768, 
                        refinement_steps: int = 1, refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describe an image using OpenAI's model with optional refinement steps.
    
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            model_name (str): The name of the OpenAI model to use.
            max_size (int): Maximum size of the image dimension before resizing.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.
    
        Returns:
            str: The generated image description text.
        """
        from openai import OpenAI
        
        client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
        
        # Initial image processing
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.resize_image(image, max_size)
        
        # Save resized image to temporary file
        image_format = image_path.split('.')[-1]
        with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
            image.save(temp_file.name)
            image_data_url = self.get_image_data_url(temp_file.name, image_format)
        
        # Initial description
        current_description = self._get_openai_description(client, image_data_url, prompt, model_name)
        
        # Refinement steps
        if refinement_steps > 1:
            template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
            
            for _ in range(refinement_steps - 1):
                refined_prompt = template.format(
                    original_query=prompt,
                    previous_answer=current_description
                )
                current_description = self._get_openai_description(client, image_data_url, refined_prompt, model_name)
        
        return current_description

    def _get_openai_description(self, client, image_data_url: str, prompt: str, model_name: str) -> str:
        """Helper method to get description from OpenAI."""
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
            model=model_name,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        return response.choices[0].message.content

    def _describe_with_groq(self, image_path: str, prompt: str, model_name: str="llama-3.2-90b-vision-preview", 
                           max_size: int = 768, refinement_steps: int = 1, 
                           refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describe an image using Groq's vision model with optional refinement steps.
    
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            model_name (str): The name of the Groq model to use.
            max_size (int): Maximum size of the image dimension before resizing.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.
    
        Returns:
            str: The generated image description text.
        """
        # Image processing
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.resize_image(image, max_size)
        
        # Save and get data URL
        image_format = image_path.split('.')[-1]
        with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
            image.save(temp_file.name)
            image_data_url = self.get_image_data_url(temp_file.name, image_format)
        
        # Initial description
        current_description = self._get_groq_description(image_data_url, prompt, model_name)
        
        # Refinement steps
        if refinement_steps > 1:
            template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
            
            for _ in range(refinement_steps - 1):
                refined_prompt = template.format(
                    original_query=prompt,
                    previous_answer=current_description
                )
                current_description = self._get_groq_description(image_data_url, refined_prompt, model_name)
        
        return current_description

    def _get_groq_description(self, image_data_url: str, prompt: str, model_name: str) -> str:
        """Helper method to get description from Groq."""
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
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        return completion.choices[0].message.content

    def _describe_gemini(self, image_path: str, prompt: str, model_name: str="gemini-1.5-flash",
                        refinement_steps: int = 1, refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describes an image using a specified Gemini model with optional refinement steps.
    
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            model_name (str): The name of the Gemini model to use.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.
        Returns:
            str: The generated image description text.
        """
        try:
            generation_config = {
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "top_k": self.config["top_k"],
                "max_output_tokens": self.config["max_tokens"],
                "response_mime_type": "text/plain",
            }

            # Load the image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            
            # Initial description
            current_description = model.generate_content([prompt, image]).text
            
            # Refinement steps
            if refinement_steps > 1:
                print("Refining the prompt for extra steps...")
                template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
                
                for _ in range(refinement_steps - 1):
                    refined_prompt = template.format(
                        previous_description=current_description
                    )
                    current_description = model.generate_content([refined_prompt, image]).text
            
            return current_description
        except Exception as e:
            return f"Error during Gemini description: {e}"

    def _describe_with_florence(self, image_path: str, prompt: str, 
                              refinement_steps: int = 1, refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describe an image using the florence2 model with optional refinement steps.
        
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.
        Returns:
            str: The generated image description text.
        """
        self.load_florence_model()
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Initial description
        current_description = self._get_florence_description(image, prompt)
        
        # Refinement steps
        if refinement_steps > 1:
            template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
            
            for _ in range(refinement_steps - 1):
                refined_prompt = template.format(
                    original_query=prompt,
                    previous_answer=current_description
                )
                current_description = self._get_florence_description(image, refined_prompt)
        
        return current_description

    def _get_florence_description(self, image: Image, prompt: str) -> str:
        """Helper method to get description from Florence model."""
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.config["max_tokens"],
            num_beams=3,
            do_sample=True,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=self.config["top_k"]
        )
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        caption = self.processor.post_process_generation(caption, task="<CAPTION>", image_size=(image.width, image.height))
        return caption["<CAPTION>"]
    
    def _describe_phi4(self, image_path: str, prompt: str, model_name: str="Phi-4-multimodal-instruct", 
                      refinement_steps: int = 1, refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describe an image using Microsoft's Phi-4 multimodal model with optional refinement steps.
    
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            model_name (str): The name of the Phi-4 model to use.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.
    
        Returns:
            str: The generated image description text.
        """
        # Create Azure client
        client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token),
        )
        
        # Get image format
        image_format = image_path.split('.')[-1].lower()
        
        # Initial description
        current_description = self._get_phi4_description(client, image_path, prompt, image_format, model_name)
        
        # Refinement steps
        if refinement_steps > 1:
            template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
            
            for _ in range(refinement_steps - 1):
                refined_prompt = template.format(
                    previous_description=current_description
                )
                current_description = self._get_phi4_description(client, image_path, refined_prompt, image_format, model_name)
        
        return current_description

    def _get_phi4_description(self, client, image_path: str, prompt: str, image_format: str, model_name: str) -> str:
        """Helper method to get description from Phi-4 model."""
        try:
            response = client.complete(
                messages=[
                    UserMessage(
                        content=[
                            TextContentItem(text=prompt),
                            ImageContentItem(
                                image_url=ImageUrl.load(
                                    image_file=image_path,
                                    image_format=image_format,
                                    detail=ImageDetailLevel.LOW)
                            ),
                        ],
                    ),
                ],
                model=model_name,
                temperature=self.config["temperature"] if self.config else 1.0,
                max_tokens=self.config["max_tokens"] if self.config else 8000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during Phi-4 description: {e}"

    def describe(self, image_path: str, prompt: Optional[str] = None, vllm_provider: str = "florence2", 
                vllm_name: Optional[str]=None, refinement_steps: int = 1, 
                refinement_prompt_path: Optional[str] = None) -> str:
        """
        Unified method to describe an image using a specified provider.

        Args:
            image_path (str): Path to the image file.
            prompt (str, optional): The prompt to guide the image description.
            vllm_provider (str, optional): The provider of the vision model.
            vllm_name (str, optional): The name of the model to use.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.

        Returns:
            str: The generated image description text.

        Raises:
            ValueError: If an unsupported `vllm_provider` is specified.
        """
        if prompt is None:
            prompt = self.prompt

        if vllm_provider == "florence2":
            return self._describe_with_florence(image_path, prompt, refinement_steps, refinement_prompt_path)
        elif vllm_provider == "groq":
            if not self.groq_client:
                raise ValueError("Groq API key not found in environment variables")
            return self._describe_with_groq(
                image_path, prompt, 
                model_name=vllm_name if vllm_name else "llama-3.2-90b-vision-preview",
                refinement_steps=refinement_steps,
                refinement_prompt_path=refinement_prompt_path
            )
        elif vllm_provider == "github":
            if vllm_name == "phi-4-multimodal-instruct":
                return self._describe_phi4(
                    image_path, prompt,
                    model_name=vllm_name,
                    refinement_steps=refinement_steps,
                    refinement_prompt_path=refinement_prompt_path
                )
            else:
                return self._describe_openai(
                    image_path, prompt, 
                    model_name=vllm_name if vllm_name else 'gpt-4o-mini',
                    refinement_steps=refinement_steps,
                    refinement_prompt_path=refinement_prompt_path
                )
        elif vllm_provider == "gemini":
            return self._describe_gemini(
                image_path, prompt, 
                model_name=vllm_name if vllm_name else "gemini-1.5-flash",
                refinement_steps=refinement_steps,
                refinement_prompt_path=refinement_prompt_path
            )
        else:
            raise ValueError(
                "Unsupported provider. Choose from 'florence2', 'groq', 'github', or 'gemini'."
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

    # Example using github with GPT-4o-mini
    start_time = time.time()
    description_github = analyzer.describe(
        image_path, prompt=prompt, vllm_provider="github", vllm_name="gpt-4o-mini"
    )
    end_time = time.time()
    print(f"\nGithub (GPT-4o-mini) Description:\n{description_github}")
    print("Github (GPT-4o-mini) process took:", end_time - start_time, "seconds.")
    
    print("#############################################")
    
    # Example using Github with Phi-4
    start_time = time.time()
    description_phi4 = analyzer.describe(
        image_path, prompt=prompt, vllm_provider="github", vllm_name="Phi-4-multimodal-instruct"
    )
    end_time = time.time()
    print(f"\nGithub (Phi-4) Description:\n{description_phi4}")
    print("Github (Phi-4) process took:", end_time - start_time, "seconds.")

