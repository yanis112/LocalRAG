from langchain_core.prompts import PromptTemplate
import streamlit as st
from src.main_utils.generation_utils_v2 import LLM_answer_v3
from abc import ABC, abstractmethod

class BaseExpert(ABC):
    """Base class for all cinematic experts.
    
    Provides common functionality for loading prompts and expert knowledge,
    and transforming prompts using specific expert knowledge.
    
    Attributes:
        model_name (str): Name of the LLM model to use
        llm_provider (str): Provider of the LLM service
    """

    def __init__(self, model_name, llm_provider,global_stylistic_guidelines=None):
        """Initialize BaseExpert with model configuration.
        
        Args:
            model_name (str): Name of the LLM model
            llm_provider (str): Provider of the LLM service
        """
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.system_prompt="You are an expert charged of image descriptions refining."
        self.global_stylistic_guidelines=global_stylistic_guidelines
        
    def load_expert_prompt(self):
        """Load expert prompt template from file.
        
        Returns:
            PromptTemplate: Loaded and formatted prompt template
        """
        with open(f"prompts/{self.prompt_file}", "r", encoding='utf-8') as f:
            template = f.read()
        return PromptTemplate.from_template(template)
    
    
    def load_expert_knowledge(self):
        """Load expert knowledge from file.
        
        Returns:
            str: Expert knowledge content
        """
        with open(f"prompts/{self.knowledge_file}", "r", encoding='utf-8') as f:
            return f.read()
    
    def transform(self, prompt):
        """Transform input prompt using expert knowledge.
        
        Args:
            prompt (str): Input prompt to transform
            
        Returns:
            str: Transformed prompt with expert knowledge applied
        """
        template = self.load_expert_prompt()
        formatted_prompt = template.format(
            prompt=prompt,
            expert_knowledge=self.load_expert_knowledge(),
            global_stylistic_guidelines=self.global_stylistic_guidelines
        )
        return LLM_answer_v3(formatted_prompt, model_name=self.model_name, 
                            llm_provider=self.llm_provider, temperature=1,system_prompt=self.system_prompt)
        
    
        
        

class RefinementExpert(BaseExpert):
    """Expert for refining general prompt structure and content."""
    prompt_file = "refinement_expert.txt"
    knowledge_file = "RefinementExpert.md"

class LightingExpert(BaseExpert):
    """Expert for enhancing lighting descriptions in prompts."""
    prompt_file = "lighting_expert.txt"
    knowledge_file = "LightingExpert.md"

class CameraAngleExpert(BaseExpert):
    """Expert for improving camera angle descriptions."""
    prompt_file = "camera_angle_expert.txt"
    knowledge_file = "CameraAngleExpert.md"

class PostProductionExpert(BaseExpert):
    """Expert for adding post-production effects to prompts."""
    prompt_file = "post_production_expert.txt"
    knowledge_file = "PostProductionExpert.md"

class FinalPromptRefiner(BaseExpert):
    """Expert for final prompt refinement and optimization."""
    prompt_file = "final_prompt_refiner.txt"
    knowledge_file = "FinalPromptRefiner.md"

class AgentCinematicExpert:
    """Main agent coordinating all cinematic experts.
    
    Manages a collection of specialized experts for prompt transformation
    and provides methods for individual or chained expert operations.
    
    Attributes:
        model_name (str): Name of the LLM model to use
        llm_provider (str): Provider of the LLM service
        experts (dict): Dictionary of available experts
    """

    def __init__(self, model_name, llm_provider,global_stylistic_guidelines=None):
        """Initialize AgentCinematicExpert with model configuration.
        
        Args:
            model_name (str): Name of the LLM model
            llm_provider (str): Provider of the LLM service
        """
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.global_stylistic_guidelines=global_stylistic_guidelines    
        self.experts = {
            'refinement': RefinementExpert(model_name, llm_provider,global_stylistic_guidelines),
            'lighting': LightingExpert(model_name, llm_provider,global_stylistic_guidelines),
            'camera': CameraAngleExpert(model_name, llm_provider,global_stylistic_guidelines),
            'post_production': PostProductionExpert(model_name, llm_provider,global_stylistic_guidelines),
            'final': FinalPromptRefiner(model_name, llm_provider)
        }
    
    def get_expert(self, expert_name):
        """Retrieve specific expert by name.
        
        Args:
            expert_name (str): Name of the expert to retrieve
            
        Returns:
            BaseExpert: The requested expert instance
        """
        return self.experts.get(expert_name)
    
    def transform_with_expert(self, prompt, expert_name):
        """Transform prompt using a specific expert.
        
        Args:
            prompt (str): Input prompt to transform
            expert_name (str): Name of the expert to use
            
        Returns:
            str: Transformed prompt
            
        Raises:
            ValueError: If expert_name is not found
        """
        expert = self.get_expert(expert_name)
        if expert:
            return expert.transform(prompt)
        raise ValueError(f"Expert {expert_name} not found")
    
    def transform_chain(self, prompt, show_progress=True):
        """Transform prompt through all experts in sequence.
        
        Args:
            prompt (str): Input prompt to transform
            show_progress (bool, optional): Whether to show progress. Defaults to True.
            
        Returns:
            str: Final transformed prompt after all experts
        """
        current_prompt = prompt
        for expert_name, expert in self.experts.items():
            if show_progress:
                st.toast(f"Refining prompt with {expert.__class__.__name__}...")
            current_prompt = expert.transform(current_prompt)
            print(f"Prompt transformed by {expert.__class__.__name__}: {current_prompt}")
            print("##############################################")
        return current_prompt
    
    def project2style(self, project_description):
        """Find the appropriate cinematic style instructions and jargon based
        on the project description (e.g genre, color palette, mood, etc.) using informations from internet.
        Args:
            project_description (str): the description of the project, what is it, e.g. A high  budget Tolkien's Silmarillion cinematic adaptation
        Returns:
            str: the cinematic style instructions and jargon
        """
        
        from src.aux_utils.internet_utils import GeminiInternetAgent
        
        #load the instruction prompt
        with open("prompts/project_description2cinematic_style.txt", "r", encoding='utf-8') as f:
            template = f.read()
        prompt = PromptTemplate.from_template(template).format(project_description=project_description)
        
        #answer using the GeminiInternetAgent
        
        agent = GeminiInternetAgent()
        
        #return the answer
        return agent.answer(prompt)

if __name__=="__main__":
    agent = AgentCinematicExpert(model_name="llama-3.3-70b-versatile", llm_provider="groq")
    prompt = """A high budget science fiction movie scene, two persons with futuristic suits are fixing an alien creature looking like
    a small grey skinned parasite shaped like a christian cross (geometric sharp angled shape) on the back of a man without t-shirt, the ceremony is important, under a deep blue magnificient sky and sun"""
    #return full refined prompt
    print("Final refined prompt:")
    print(agent.transform_chain(prompt))