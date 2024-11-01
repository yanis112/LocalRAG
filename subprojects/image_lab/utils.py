from src.generation_utils_v2 import LLM_answer_v3
import os
import shutil
from huggingface_hub import login, whoami
import streamlit as st

# List of Hugging Face tokens
hf_token_list = [
    "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", #deep trader 2
    "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", #deep trader
    "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"
]

# Define input and output directories
INPUT_DIR = 'subprojects/image_lab/input_images'
OUTPUT_DIR = 'subprojects/image_lab/output_images'

#Model setup
MODEL_NAME= 'gpt-4o-mini' #'llama-3.2-90b-text-preview'
LLM_PROVIDER= 'github'

# Create input and output directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def enhance_image(input_path: str) -> str:
    """
    Enhances an image using the finegrain-image-enhancer model.

    Args:
        input_path (str): Path to the input image.

    Returns:
        str: Path to the enhanced image.
    """
    from gradio_client import Client, handle_file
    for token in hf_token_list:
        try:
            # Login with the current token
            login(token=token)
            
            # Verify login status
            user_info = whoami()
            
            # Initialize the client with the current token
            client = Client(
                "finegrain/finegrain-image-enhancer",
                hf_token=token
            )
            
            # Call the client.predict method with handle_file
            result = client.predict(
                input_image=handle_file(input_path),
                prompt="high quality, high resolution, stunning lighting, cinematic look",
                negative_prompt="bad quality, low resolution, blurry",
                seed=42,
                upscale_factor=3,
                controlnet_scale=0.8,
                controlnet_decay=1,
                condition_scale=6,
                tile_width=200,
                tile_height=200,
                denoise_strength=0.3,
                num_inference_steps=30,
                solver="DDIM",
                api_name="/process"
            )
            
            # Check if the result is valid
            if result and isinstance(result, list) and len(result) > 1:
                enhanced_image_path = result[1]  # Select the second path as the enhanced image
                filename = os.path.basename(input_path)
                output_filename = f"enhanced_{filename}"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # Copy the enhanced image to the output directory
                shutil.copy(enhanced_image_path, output_path)
                return output_path
        except Exception as e:
            continue
    
    return None

def analyze_image(image):
    from src.image_analysis import ImageAnalyzer
    image_path = os.path.join(INPUT_DIR, "temp_image.jpg")
    with open(image_path, "wb",encoding='utf-8') as f:
        f.write(image.getbuffer())
    
    enhanced_image_path = enhance_image(image_path)
    if enhanced_image_path:
        image_path = enhanced_image_path
    
    analyzer = ImageAnalyzer()
    result = analyzer.describe_advanced(image_path=image_path, prompt="List with extreme precision all the base ingredients in this photograph, don't forget any.", grid_size=2)
    return result

# def refine_prompt(prompt, style_prompt=None):
#     final_prompt = f"""Here is an image description for a text-to-image generation model given by a user: {prompt}. Refine the prompt to make it more suitable for the model. For this
#     you will make it clearer, better written, with correct gramar. The prompt should be 5-10 sentences long and highly detailed, only visual elements and camera angles, light effects should be described, no atmoshpere or emotions or intanangible elements, the prompt should't contain any proper nouns of places, people, ect.. but only precise visual descriptions of those elements. Answer without preamble."""
    
#     if style_prompt:
#         final_prompt = f"{final_prompt}. You will refine the current prompt by adding effects / fixing camera angles such as described in the following guide: {style_prompt}"

#     recipe = LLM_answer_v3(final_prompt, model_name="llama3-405b", llm_provider="sambanova", temperature=1)
#     return recipe



class RefinementExpert:
    def load_expert_prompt(self):
        #load the expert prompt from subprojects/image_lab/RefinementExpert.md
        with open("subprojects/image_lab/RefinementExpert.md", "r",encoding='utf-8') as f:
            expert_prompt = f.read()
        return expert_prompt
    
    def transform(self, prompt):
        #load the expert prompt
        expert_prompt = self.load_expert_prompt()
        final_prompt = f"""Here is a single paragraph image description given by a user as input \
        for a text-to-image generation model, in the intent to generate \
        a movie/film still:  <prompt_start> {prompt} <prompt_end>. Refine the image description to \
        make it clearer, better written, with correct grammar. The prompt should be 5-10 sentences long and \
        highly detailed. You will use for this task the following expert knowledge: {expert_prompt} Answer \
        the one-paragraph image description without preamble, and ensure all the original visual elements of the description are preserved."""
        return LLM_answer_v3(final_prompt, model_name=MODEL_NAME, llm_provider=LLM_PROVIDER, temperature=1)

class LightingExpert:
    def load_expert_prompt(self):
        # Load the expert prompt from subprojects/image_lab/LightingExpert.md
        with open("subprojects/image_lab/LightingExpert.md", "r",encoding='utf-8') as f:
            expert_prompt = f.read()
        return expert_prompt
    
    def transform(self, prompt):
        # Load the expert prompt
        expert_prompt = self.load_expert_prompt()
        final_prompt = f"""Here is a single paragraph image description given by a user as input for a text-to-image generation model,\
        in the intent to generate a movie/film still: \
        <prompt_start> {prompt} <prompt_end>. Refine the image description by adding detailed descriptions of lighting and light effects.\
        Make sure to specify the type of lighting, its source, and its impact on the scene. You will use for this task the following expert knowledge: {expert_prompt}\
        Answer the one-paragraph image description without preamble, and ensure all the original visual elements of the description are preserved."""
        return LLM_answer_v3(final_prompt, model_name=MODEL_NAME, llm_provider=LLM_PROVIDER, temperature=1)


class CameraAngleExpert:
    def load_expert_prompt(self):
        # Load the expert prompt from subprojects/image_lab/CameraAngleExpert.md
        with open("subprojects/image_lab/CameraAngleExpert.md", "r",encoding='utf-8') as f:
            expert_prompt = f.read()
        return expert_prompt
    
    def transform(self, prompt):
        # Load the expert prompt
        expert_prompt = self.load_expert_prompt()
        final_prompt = f"""Here is a single paragraph image description given by a user as input for a text-to-image generation model,\
        in the intent to generate a movie/film still: <prompt_start> {prompt} <prompt_end>. Improve the image description by specifying an unique professional camera angle for the scene.\
        You will use for this task the following expert knowledge: {expert_prompt}.\
        Answer the enhanced one-paragraph image description without preamble, and ensure all the original visual elements of the description are preserved."""
        return LLM_answer_v3(final_prompt, model_name=MODEL_NAME, llm_provider=LLM_PROVIDER, temperature=1)


class PostProductionExpert:
    def load_expert_prompt(self):
        # Load the expert prompt from subprojects/image_lab/PostProductionExpert.md
        with open("subprojects/image_lab/PostProductionExpert.md", "r",encoding='utf-8') as f:
            expert_prompt = f.read()
        return expert_prompt
    
    def transform(self, prompt):
        # Load the expert prompt
        expert_prompt = self.load_expert_prompt()
        final_prompt = f"""Here is a single paragraph image description given by a user as input for a text-to-image generation model, \
        in the intent to generate a movie/film still: <prompt_start> {prompt} <prompt_end>. Enhance the image description by adding post-production effects and cinematic enhancements. \
        Specify any filters, color grading, and visual effects that should be applied to make the description better (consistent with the visual elements). \
        You will use for this task the following expert knowledge: {expert_prompt} \
        Answer the enhanced one-paragraph image description without preamble, and ensure all the original visual elements of the description are preserved."""
        return LLM_answer_v3(final_prompt, model_name=MODEL_NAME, llm_provider=LLM_PROVIDER, temperature=1)

class FinalPromptRefiner:
    def load_expert_prompt(self):
        # Load the expert prompt from subprojects/image_lab/FinalPromptRefiner.md
        with open("subprojects/image_lab/FinalPromptRefiner.md", "r",encoding='utf-8') as f:
            expert_prompt = f.read()
        return expert_prompt
    
    def transform(self, prompt):
        # Load the expert prompt
        expert_prompt = self.load_expert_prompt()
        final_prompt = f"""Here is an image description given by a user as input for a text-to-image generation model: <prompt_start> {prompt} <prompt_end>. \
        Shorten the image description by approximately 20% while ensuring it remains clear, concise, and coherent in its choices of lighting, camera angles, etc...\
        You will use for this task the following expert knowledge: {expert_prompt} Answer the refined image description without preamble, 
        and ensure all the original visual elements of the description are preserved."""
        return LLM_answer_v3(final_prompt, model_name=MODEL_NAME, llm_provider=LLM_PROVIDER, temperature=1)


def refine_prompt(prompt):
    experts = [RefinementExpert(), LightingExpert(), CameraAngleExpert(), PostProductionExpert(), FinalPromptRefiner()]
    for expert in experts:
        st.toast(f"Refining prompt with {expert.__class__.__name__}...")
        prompt = expert.transform(prompt)
        print(f"Prompt transformed by expert {expert.__class__.__name__}: {prompt}")
        print("##############################################")
    return prompt

def prompt2video(prompt):
    final_prompt = f"""Here is an image description for a text-to-image generation model given by a user, in the intent to generate a movie/film still:  <prompt_start> {prompt} <prompt_end>. Based on this description, you will make a prompt for an image-to-video generation model whose goal is to animate the scene described in the image in a cinematic way.
    The prompt should be very concise, precising camera type of movement, characters and their actions, and the setting. The prompt should be only one or two sentence long. Answer the prompt without preamble."""

    recipe = LLM_answer_v3(final_prompt, model_name="gpt-4o", llm_provider="github")
    return recipe

def prompt2sound(prompt):
    final_prompt = f"""Here is an image description for a text-to-image generation model given by a user, in the intent to generate a movie/film still: <prompt_start> {prompt} <prompt_end>. Based on this description, you will make a prompt for an image-to-sound generation model. The goal of this model is to generate a cinematic sound that would fit the scene described in the image.
    The prompt should be very concise, precising the type of sound that fits the scene. The prompt should be only one short sentence. Answer the prompt without preamble."""

    recipe = LLM_answer_v3(final_prompt, model_name="gpt-4o", llm_provider="github")
    return recipe