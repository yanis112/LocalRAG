import gradio_client
from typing import Dict, Any
import json

class WebVisionAgent:
    def __init__(self, box_threshold: float = 0.05, iou_threshold: float = 0.1):
        """Initialize the WebVisionAgent"""
        self.box_threshold = box_threshold
        self.iou_threshold = iou_threshold
        self.client = gradio_client.Client("microsoft/OmniParser")

    def analyze_page(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze page components from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with component information
        """
        # Call OmniParser API
        result = self.client.predict(
            image_path,
            self.box_threshold,
            self.iou_threshold,
            api_name="/process"
        )
        
        # Parse coordinates from result[2] (coordinates string)
        coords = json.loads(result[2])
        
        # Structure components
        components = {}
        
        # Parse text boxes from result[1] (parsed elements string)
        text_elements = result[1].split('\n')
        
        for line in text_elements:
            if line.startswith('Text Box ID '):
                id_str = line.split(':')[0].replace('Text Box ID ', '')
                text = line.split(':')[1].strip()
                
                if id_str in coords:
                    components[f"text_{id_str}"] = {
                        "type": "text",
                        "content": text,
                        "coordinates": {
                            "x": coords[id_str][0],
                            "y": coords[id_str][1],
                            "width": coords[id_str][2],
                            "height": coords[id_str][3]
                        }
                    }
            elif line.startswith('Icon Box ID '):
                id_str = line.split(':')[0].replace('Icon Box ID ', '')
                icon = line.split(':')[1].strip()
                
                if id_str in coords:
                    components[f"icon_{id_str}"] = {
                        "type": "icon",
                        "content": icon,
                        "coordinates": {
                            "x": coords[id_str][0],
                            "y": coords[id_str][1],
                            "width": coords[id_str][2],
                            "height": coords[id_str][3]
                        }
                    }
                    
        return components
    
if __name__ == "__main__":
    # Initialize agent
    import time
    agent = WebVisionAgent()

    # Analyze image
    start_time = time.time()
    components = agent.analyze_page("aux_data/test_web_screenshot.png")

    # Access components
    for comp_id, details in components.items():
        print(f"{comp_id}:")
        print(f"  Type: {details['type']}")
        print(f"  Content: {details['content']}")
        print(f"  Position: {details['coordinates']}")
    end_time = time.time()
    print("Time taken:", end_time - start_time)