import os
import time
from pathlib import Path
from typing import Optional, Tuple

from langchain_core.prompts import PromptTemplate
from src.aux_utils.notion_utils import NotionAgent
from src.main_utils.generation_utils_v2 import LLM_answer_v3
from src.main_utils.vectorstore_utils_v5 import VectorAgent

class MeetingAgent:
    """Agent responsible for handling meeting-related tasks like summarization, Notion integration, and vector store updates"""
    
    def __init__(self, config: dict, qdrant_client=None):
        """Initialize the MeetingAgent with configuration and optional Qdrant client.
        
        Args:
            config (dict): Configuration dictionary containing model settings
            qdrant_client: Optional Qdrant client for vector store operations
        """
        self.config = config
        self.qdrant_client = qdrant_client
        self.notion_agent = NotionAgent()
        
        # Load the meeting summary prompt template
        with open("prompts/meeting_summary_prompt.txt", "r", encoding="utf-8") as f:
            self.summary_prompt = f.read()
        self.summary_prompt_template = PromptTemplate.from_template(self.summary_prompt)

    def generate_summary(self, transcription: str, user_query: str = "") -> str:
        """Generate a structured summary of the meeting transcription.
        
        Args:
            transcription (str): The raw meeting transcription
            user_query (str): Optional specific user requirements for the summary
            
        Returns:
            str: Formatted meeting summary in markdown
        """
        # Format the prompt with transcription and user query
        full_prompt = self.summary_prompt_template.format(
            transcription=transcription,
            user_query=user_query
        )
        
        # Generate summary using the configured LLM
        summary = LLM_answer_v3(
            prompt=full_prompt,
            stream=False,
            model_name=self.config["model_name"],
            llm_provider=self.config["llm_provider"],
            temperature=self.config.get("temperature", 1.0)
        )
        
        return summary

    def save_to_notion(self, summary: str) -> str:
        """Create a Notion page from the meeting summary.
        
        Args:
            summary (str): The markdown formatted meeting summary
            
        Returns:
            str: ID of the created Notion page
        """
        # Extract title from first line of summary (assumes markdown format)
        title = summary.split("\n")[0].lstrip("#").strip()
        
        # Create Notion page
        page_id = self.notion_agent.create_page_from_markdown(
            markdown_content=summary,
            page_title=title
        )
        
        return page_id

    def save_to_file(self, summary: str) -> str:
        """Save the meeting summary to a local file.
        
        Args:
            summary (str): The meeting summary content
            
        Returns:
            str: Path to the saved file
        """
        # Create meeting_summary directory if it doesn't exist
        Path("data/meeting_summary").mkdir(parents=True, exist_ok=True)
        
        # Create filename with current date
        filename = f"data/meeting_summary/{time.strftime('%Y-%m-%d')}.txt"
        
        # Write summary to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Meeting summary for the date: {time.strftime('%Y-%m-%d')}\n\n")
            f.write(summary)
        
        return filename

    def update_vectorstore(self) -> None:
        """Update the vector store with new meeting summaries."""
        if not self.qdrant_client:
            raise ValueError("Qdrant client must be provided to update vector store")
            
        vector_agent = VectorAgent(
            default_config=self.config,
            qdrant_client=self.qdrant_client
        )
        vector_agent.fill()

    def process_meeting(self, transcription: str, user_query: str = "") -> Tuple[str, str, Optional[str]]:
        """Process a meeting transcription end-to-end: generate summary, save to Notion and file, update vector store.
        
        Args:
            transcription (str): The raw meeting transcription
            user_query (str): Optional specific requirements for the summary
            
        Returns:
            Tuple containing:
            - str: Generated summary
            - str: Path to saved file
            - str: Notion page ID (if successful) or None
        """
        # Generate summary
        summary = self.generate_summary(transcription, user_query)
        
        # Save to file
        file_path = self.save_to_file(summary)
        
        # Save to Notion
        try:
            notion_page_id = self.save_to_notion(summary)
        except Exception as e:
            print(f"Failed to save to Notion: {e}")
            notion_page_id = None
            
        # Update vector store
        try:
            self.update_vectorstore()
        except Exception as e:
            print(f"Failed to update vector store: {e}")
            
        return summary, file_path, notion_page_id