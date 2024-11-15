import os
import pandas as pd
from src.image_analysis import ImageAnalyzer
from tqdm import tqdm

class InstagramAgent:
    def __init__(self, image_source_folder):
        self.image_source_folder = image_source_folder
        self.persistence_dir = "./subprojects/instagram_gestion"
        self.csv_path = os.path.join(self.persistence_dir, 'instagram_data.csv')
        self.image_analyzer = ImageAnalyzer(model_name="gpt-4o")
        self.prompt ="Describe this image in a very detailled way, including poses, objects, and actions."
        
        # Create persistence directory if it doesn't exist
        os.makedirs(self.persistence_dir, exist_ok=True)
        
        # Load existing dataframe or create new one
        if os.path.exists(self.csv_path):
            self.dataframe = pd.read_csv(self.csv_path)
        else:
            self.dataframe = pd.DataFrame(columns=['path', 'description', 'already_published', 'instagram_caption'])
    
    def save_dataframe(self):
        """Save the dataframe to CSV file"""
        self.dataframe.to_csv(self.csv_path, index=False)
    
    def show_dataframe(self):
        print(self.dataframe)
    
    def describe_all(self):
        image_paths = [os.path.join(self.image_source_folder, f) 
                      for f in os.listdir(self.image_source_folder) 
                      if os.path.isfile(os.path.join(self.image_source_folder, f))]
        
        print("Images paths: ", image_paths)
        
        # Filter out already processed images
        new_images = [path for path in image_paths 
                     if os.path.abspath(path) not in self.dataframe['path'].values]
        
        for image_path in tqdm(new_images, desc='Describing new images...'):
            description = self.image_analyzer.describe_with_groq(image_path, prompt=self.prompt,grid_size=1)
            #describe_advanced(image_path, prompt=self.prompt,grid_size=1)
            #describe_with_groq(image_path, prompt=self.prompt,grid_size=1)
            data = {
                'path': os.path.abspath(image_path),
                'description': description,
                'already_published': False,
                'instagram_caption': None
            }
            self.dataframe = pd.concat([self.dataframe, 
                                      pd.DataFrame([data])], 
                                      ignore_index=True)
            # Save after each new image is processed
            self.save_dataframe()
            
            
if __name__ == '__main__':
    agent = InstagramAgent(image_source_folder='./subprojects/instagram_gestion/image_source')
    agent.describe_all()
    agent.show_dataframe()