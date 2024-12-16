from src.aux_utils.vision_utils_v2 import ImageAnalyzerAgent
from src.main_utils.generation_utils import LLM_answer_v3



def analyze_image(image):
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(image.getbuffer())
    
    analyzer = ImageAnalyzerAgent()
    result = analyzer.describe(image_path=image_path, prompt="List with extreme precision all the base ingredients in this photograph, don't forget any.", grid_size=2)
    return result


def generate_recipe(ingredients):
    prompt = f"Create a nice cooking recipe using some of the following ingredients list, you don't have the right to use ingredients that are not in the list exept for spices: {ingredients}. # You will answer in french."
    recipe = LLM_answer_v3(prompt, model_name="gpt-4o", llm_provider="github")
    return recipe