# -*- coding: utf-8 -*-
"""Module for handling job-related content generation and processing."""

from langchain_core.prompts import PromptTemplate
import json
import yaml
import streamlit as st
from src.main_utils.utils import extract_url
from src.main_utils.link_gestion import ExternalKnowledgeManager
from src.main_utils.generation_utils_v2 import LLM_answer_v3
import re
import logging


class JobWriterAgent:
    """Agent responsible for generating job application content and processing job-related queries."""

    def __init__(self, config):
        """Initialize the JobWriterAgent.

        Args:
            config: Configuration dictionary containing model and provider settings.
        """
        self.config = config
        self.knowledge_manager = ExternalKnowledgeManager(config)
        
        # Set up logging similar to generation utils
        self.logger = logging.getLogger("JobAgentV2")
        if not self.logger.handlers:
            handler = logging.FileHandler("logging/job_agent.log")
            formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

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
        """Reformulate prompt with extracted resource using a prompt file and LangChain's PromptTemplate.
        
        This method takes the user query and replaces the URL with the scrapped job offer content in markdown format,
        removing unnecessary website noise.
        
        Args:
            original_query: Original user query containing a URL.
            resource: Extracted job offer markdown content.
        
        Returns:
            str: Reformulated prompt.
        """
        if resource:
            with open("prompts/job_offer_parser_prompt.txt", "r", encoding="utf-8") as f:
                parser_prompt = f.read()
            template = PromptTemplate.from_template(parser_prompt)
            formatted_prompt = template.format(original_query=original_query, job_offer=resource)
            
            with st.spinner("Processing job offer parser prompt...", show_time=True):
                result = LLM_answer_v3(
                    prompt=formatted_prompt,
                    stream=False,
                    model_name=self.config["job_offer_parser_model_name"],
                    llm_provider=self.config["job_offer_parser_llm_provider"],
                    system_prompt=(
                        "You are a job offer parser that extracts the job offer in markdown format. "
                        "Replace the URL in the query with the provided job offer content, ensuring that only the "
                        "essential job offer details are kept, without the extraneous website information."
                    )
                )
            return result
        return original_query

    def reformulate_prompt_v2(self, original_query, offer_description=None):
        """Simpler method to reformulate prompt by removing URLs using regex and adding offer description.
        
        Args:
            original_query: Original user query containing URLs.
            offer_description: Offer description text to insert between separators.
        
        Returns:
            str: Reformulated prompt.
        """
        self.logger.info("Starting simpler reformulation of prompt.")
        pattern = r"http[s]?://\S+"
        modified_query = re.sub(pattern, "", original_query).strip()
        self.logger.info("Removed URL from query. Original: '%s', after removal: '%s'", original_query, modified_query)
        if offer_description:
            modified_query = f"{modified_query}\n=== OFFER DESCRIPTION START ===\n{offer_description}\n=== OFFER DESCRIPTION END ==="
            self.logger.info("Added offer description to query.")
        self.logger.info("Final reformulated prompt: %s", modified_query)
        return modified_query
    
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
            self.logger.info("Extracted resource from URL: %s", resource)
            query = self.reformulate_prompt(original_query=query, resource=resource)
        else:
            self.logger.info("No URL found in query.")
            # Alternatively, you could call the original method if needed
            # query = self.reformulate_prompt(query)

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

