# The goal of this file is to manage the links given by the user to extract the associate rescource and save it in the database
import re
import streamlit as st
import asyncio
from crawl4ai.crawl4ai import AsyncWebCrawler #Replace this with  just crawl4ai in the final version !!!!


def extract_linkedin(link):
    """
    Takes a Linkedin post link as input and extracts the content of the post.

    Args:
        link (str): The URL of the Linkedin post.

    Returns:
        str: The extracted content of the Linkedin post.
    """
    
    print("Link given to the extract_linkedin function: ",link)
    #we must reformart the link if not formatted correctly (remove spaces at thebeginning or end and extract it using regex)
    link = link.strip()
    
    
    async def main(url):
        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                result = await crawler.arun(url=url)
                return result.markdown
        except Exception as e:
            print(f"Error extracting LinkedIn content: {e}")
            return None

    # Run the async function
    content = asyncio.run(main(link))
    
    #convert the content to str
    content=str(content)
    
    return content


class ExternalKnowledgeManager():
    def __init__(self,config,client=None):
        self.current_transcription=None
        self.config=config
        self.client=client
    
    def classify_rescource(self,link):
        """
        Classify the given link as a specific type of resource.
        This method checks the provided link and classifies it as either a 
        YouTube video, a LinkedIn profile, or a general website based on 
        the content of the link.
        Args:
            link (str): The URL or link to be classified.
        Returns:
            str: The type of resource the link represents. Possible values 
            are 'video' for YouTube videos, 'linkedin' for LinkedIn profiles, 
            and 'website' for general websites.
        """
        
        if 'youtu' in str(link):
            print("This is a youtube video !")
            return 'video'
        elif 'linkedin' in str(link):
            print("This is a linkedin post !")
            return 'linkedin'
        else:
            return 'website'
        
    def extract_rescource(self,link):
        """
        Extracts the resource from the given link based on its classification.
        This method classifies the provided link and extracts the resource accordingly.
        If the link is classified as a video, it uses YouTubeTranscriber to transcribe the video.
        If the link is classified as a LinkedIn resource, it extracts the LinkedIn resource.
        Otherwise, it extracts the resource from a website.
        Args:
            link (str): The URL of the resource to be extracted.
        Returns:
            str: The extracted transcription if the resource is a video, or the extracted content from LinkedIn or a website.
        """
        
        with st.spinner("Extracting the rescource..."):
            if self.classify_rescource(link)=='video':
                from src.aux_utils.transcription_utils import YouTubeTranscriber
                transcriber=YouTubeTranscriber()
                transcription=transcriber.transcribe(input_path=link,method="groq")
                self.current_rescource="### This is a video transcription ### "+ transcription
                print("Rescource extracted successfully !")
                return transcription
                
            elif self.classify_rescource(link)=='linkedin':
                rescource=str(extract_linkedin(link))    
                print("FULL RESCOURCE: ",rescource)
                self.current_rescource="### This is a linkedin post ### "+ rescource
                print("Rescource extracted successfully !")
                return rescource
            else:
                return self.extract_website(link)
        
    def index_rescource(self):
        """
        Indexes the current transcription as a resource.
        This method performs the following steps:
        1. Classifies the current transcription to determine its topic.
        2. Saves the transcription in a Markdown file in the appropriate directory based on the topic.
        3. Indexes the saved resource in the database.
        The directory for saving the resource is determined by the topic classification, 
        which maps to a directory specified in the configuration.
        Imports:
            time: Standard library module for time-related functions.
            IntentClassifier: Class for classifying text into predefined topics.
            VectorAgent: Class for handling vector-based operations in the database.
        Raises:
            Any exceptions raised during file operations or classification will propagate up.
        Side Effects:
            Creates a new Markdown file in the appropriate directory.
            Prints a success message upon saving the resource.
        """
        
        import time
        # First we save the rescource in a file in the appropriate folder
        from src.aux_utils.text_classification_utils import IntentClassifier
        with st.spinner("Indexing the rescource..."):
            
            topic_classifier=IntentClassifier(labels=list(self.config['data_sources'].values()))
            
            #get the associated topic description
            topic=topic_classifier.classify(self.current_rescource)
            
            print("Topic Finded by classifier: ",topic)
            
            #get the directory associated to the topic (the key associated to the topic in the dictionary)
            directory=[key for key, value in self.config['data_sources'].items() if value == topic][0]
            
            #save the rescource in the data/{directory} folder in .md format
            
            with open(f'data/{directory}/rescource_{time.time()}.md','w',encoding='utf-8') as file:
                file.write(self.current_rescource)
                print("Rescource saved successfully ! in the data/"+directory+" folder !")
            
            #we index the rescource in the database !
            from src.main_utils.vectorstore_utils_v2 import VectorAgent
            #check if qdrant client is in session state
            
            vector_agent = VectorAgent(default_config=self.config,qdrant_client=self.client)
            vector_agent.fill()
               
               
if __name__=="__main__":
   
    #load config using yaml from config/config.yaml
    # import yaml
    # with open('config/config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    # manager=ExternalKnowledgeManager(config)
    # link="https://youtu.be/_4ZoBEmcFXI?si=LLManhdxYnzJMc7K"
    # manager.extract_rescource(link)
    # manager.index_rescource()
    link="https://www.linkedin.com/posts/yanis-labeyrie-67b11b225_openai-gpt5-o1-activity-7240116943750385664-uxzS?utm_source=share&utm_medium=member_desktop"
    content=extract_linkedin(link)
    print("Type of the content: ",type(content))
    print("##############################################")
    print(content)
    print("##############################################")