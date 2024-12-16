from src.vision_utils_v2 import ImageAnalyzerAgent

result = ImageAnalyzerAgent().describe_advanced(image_path="test_images/fridge.jpg",prompt="List with extreme precision all the base ingredients in this photograph, don't forget any.",grid_size=2)
print("Result:",result)