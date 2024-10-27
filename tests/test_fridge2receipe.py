from src.image_analysis import ImageAnalyzer

result = ImageAnalyzer().describe_advanced(image_path="test_images/fridge.jpg",prompt="List with extreme precision all the base ingredients in this photograph, don't forget any.",grid_size=2)
print("Result:",result)