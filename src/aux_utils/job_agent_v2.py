
from weakref import ref
from src.main_utils.utils import extract_url
from langchain_core.prompts import PromptTemplate
import json
import yaml
import streamlit as st
from src.main_utils.link_gestion import ExternalKnowledgeManager
from src.main_utils.generation_utils_v2 import LLM_answer_v3

class JobWriterAgent:
    def __init__(self, config):
        self.config = config
        self.knowledge_manager = ExternalKnowledgeManager(config)
    
    
    def extract_resource(self, url):
        """Extract resource content from URL"""
        return self.knowledge_manager.extract_rescource_from_link(url)
    
    def reformulate_prompt(self, original_query, resource=None):
        """Reformulate prompt with extracted resource
        Args:
            original_query (str): Original user query
            resource (str): Extracted resource content
        Returns:
            str: Reformulated prompt
        """
        if resource:
            system_prompt= "You are a prompt engineer in charge of reformulating prompts for a language model."
            formulating_prompt= f"""Reformulate the following prompt by replacing the URL with the formatted content 
                of the website associated with appropriate delimitors, you can reformulate better the user  
                query to make it better written without changing the main idea/goal. Here is the original prompt: \n\n{original_query} and
                here is the content extracted from the URL: \n\n{resource}, return the reformulated / restructured prompt without preamble. """
            
            #get LLM answer
            with st.spinner("Formulating the query for the agent..."):
                reformulated_query = LLM_answer_v3(
                    prompt=formulating_prompt,
                    stream=False,
                    model_name=self.config["model_name"],
                    llm_provider=self.config["llm_provider"],
                    system_prompt=system_prompt
                )
                
        return reformulated_query
        
    def generate_content(self, query):
        """Takes the user query that want to write to an employer
        and find out if there is a job offer or company description url in the query.
        if there is, it extracts the content from the url and reformulate the prompt with the content.
        
        Args:
            query (str): User query
        Returns:
            str: Generated response
        """
        
        
        #check for url in the user query
        url = extract_url(query) #returns None if no url is found and url if found
        resource = self.extract_resource(url) #extract the resource content from the url
        query = self.reformulate_prompt(query, resource) #reformulate the prompt (instruction + url) replacing the url with the content
        print("Reformulated prompt: ", query)  #print the reformulated prompt
        
            
        # Load templates and data
        with open("prompts/company_contact_prompt.txt", "r", encoding='utf-8') as f:
            template = f.read()
        template = PromptTemplate.from_template(template)
        
        info_dict = json.dumps(yaml.safe_load(open("aux_data/info.yaml", "r")))
        cv = json.dumps(open("aux_data/cv.txt", "r").read())
        
        # Format prompt
        full_prompt = template.format(query=query, infos=info_dict, cv=cv)
        
        # Generate response
        system_prompt = """You are a job application assistant in charge of writing messages to potential employers."""
        
        with st.spinner("Generating a text for a job offer..."):
            answer = LLM_answer_v3(
                prompt=full_prompt,
                stream=True,
                model_name=self.config["model_name"],
                llm_provider=self.config["llm_provider"],
                system_prompt=system_prompt
            )
            
        return answer, []
        
       