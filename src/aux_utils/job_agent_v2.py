"""Module for handling job-related content generation and processing."""

from langchain_core.prompts import PromptTemplate
import json
import yaml
import streamlit as st
from src.main_utils.utils import extract_url
from src.main_utils.link_gestion import ExternalKnowledgeManager
from src.main_utils.generation_utils_v2 import LLM_answer_v3


class JobWriterAgent:
    """Agent responsible for generating job application content and processing job-related queries."""

    def __init__(self, config):
        """Initialize the JobWriterAgent.

        Args:
            config: Configuration dictionary containing model and provider settings.
        """
        self.config = config
        self.knowledge_manager = ExternalKnowledgeManager(config)
    
    def extract_resource(self, url):
        """Extract resource content from URL.

        Args:
            url: The URL to extract content from.

        Returns:
            str: The extracted content.
        """
        return self.knowledge_manager.extract_rescource_from_link(url)
    
    def read_motivation_intro(self, intro_path="aux_data/amorce_motivation.txt"):
        """Read the motivation letter introduction from a file.

        Args:
            intro_path: Path to the file containing the introduction text.

        Returns:
            str: The introduction text.
        """
        try:
            with open(intro_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def reformulate_prompt(self, original_query, resource=None):
        """Reformulate prompt with extracted resource.

        Args:
            original_query: Original user query.
            resource: Extracted resource content.

        Returns:
            str: Reformulated prompt.
        """
        if resource:
            system_prompt = "You are a prompt engineer reformulating prompts for a language model."
            formulating_prompt = (
                f"Reformulate this prompt by replacing the URL with the formatted content "
                f"of the website. Improve the writing while keeping the main idea. "
                f"Original prompt:\n\n{original_query}\n\n"
                f"Content from URL:\n\n{resource}\n\n"
                f"Return the reformulated prompt without preamble."
            )
            
            with st.spinner("Formulating the query for the agent...", show_time=True):
                reformulated_query = LLM_answer_v3(
                    prompt=formulating_prompt,
                    stream=False,
                    model_name=self.config["model_name"],
                    llm_provider=self.config["llm_provider"],
                    system_prompt=system_prompt
                )
                
        return reformulated_query
        
    def generate_content(self, query):
        """Generate job application content based on user query.

        Takes a user query about writing to an employer, extracts any URLs, reformulates
        the prompt if needed, and generates appropriate content.

        Args:
            query: User query string.

        Returns:
            tuple: Generated response and empty list (for compatibility).
        """
        url = extract_url(query)
        if url is not None:
            resource = self.extract_resource(url)
            query = self.reformulate_prompt(query, resource)
        print("Reformulated prompt: ", query)
        
        # Load templates and data
        with open("prompts/company_contact_prompt.txt", "r", encoding="utf-8") as f:
            template = f.read()
        template = PromptTemplate.from_template(template)
        
        info_dict = json.dumps(yaml.safe_load(open("aux_data/info.yaml", "r")))
        cv = json.dumps(open("aux_data/cv.txt", "r").read())
        amorce = self.read_motivation_intro()
        
        # Format prompt with amorce
        full_prompt = template.format(query=query, infos=info_dict, cv=cv, amorce=amorce)
        
        # Generate response
        system_prompt = "You are a job application assistant for potential employers."
        
        with st.spinner("Generating a text for a job offer...", show_time=True):
            answer = LLM_answer_v3(
                prompt=full_prompt,
                stream=True,
                model_name=self.config["model_name"],
                llm_provider=self.config["llm_provider"],
                system_prompt=system_prompt
            )
            
        return answer, []

