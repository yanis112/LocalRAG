import os
from time import sleep

import pandas as pd
from httpx import stream
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from src.main_utils.generation_utils_v2 import LLM_answer_v3


class JobAgent:
    """
    A class used to represent a Job Agent that scrapes job listings from various websites.
    """
    def __init__(self,config, is_remote: bool = False, qdrant_client=None):
        """
        Initialize the JobAgent with LLM configuration.

        Args:
            model_name (str): Name of the LLM model to use
            llm_provider (str): Provider of the LLM (e.g., 'openai', 'anthropic')
            is_remote (bool): Whether to search for remote jobs
            qdrant_client: Qdrant client instance for RAG operations
        """
        self.config=config
        self.model_name = config["model_name"]
        self.llm_provider = config["llm_provider"]
        self.is_remote = is_remote
        self.qdrant_client = qdrant_client

    @staticmethod
    @tool(return_direct=True)
    def scrape_and_convert(search_terms: list[str], locations: list[str], hours_old: int = 200, 
                          results_wanted: int = 40, is_remote: bool = False) -> str:
        """
        Scrape job listings and convert them to markdown files.

        Args:
            search_terms (list[str]): List of job titles or keywords to search for
            locations (list[str]): List of locations to search in
            hours_old (int, optional): Maximum age of listings in hours
            results_wanted (int, optional): Number of results to fetch
            is_remote (bool, optional): Whether to search for remote jobs

        Returns:
            str: Success or error message
        """
        from jobspy import scrape_jobs
        
        try:
            # Calculate results per combination
            total_combinations = len(search_terms) * len(locations)
            results_per_search = max(1, results_wanted // total_combinations)
            
            all_jobs = []

            # Create google search terms by combining search terms with locations
            google_search_terms = [
                f"{term} {loc}" for term in search_terms for loc in locations
            ]
            
            # Fetch jobs using jobspy
            for search_term, google_term in zip(search_terms * len(locations), google_search_terms):
                for location in locations:
                    try:
                        jobs = scrape_jobs(
                            site_name=["indeed", "linkedin", "glassdoor", "google"],
                            google_search_term=google_term,
                            search_term=search_term,
                            location=location,
                            results_wanted=results_per_search,
                            hours_old=hours_old,
                            country_indeed='france',
                            is_remote=is_remote
                        )
                        if jobs is not None:
                            all_jobs.append(jobs)
                        sleep(1)  # Rate limiting
                    except Exception as e:
                        print(f"Error fetching jobs for {search_term} in {location}: {str(e)}")
                        continue

            if not all_jobs:
                return "No jobs found matching your criteria."
                
            # Combine all results
            combined_jobs = pd.concat(all_jobs, ignore_index=True)
            
            # Create directory if needed
            if not os.path.exists('data/jobs'):
                os.makedirs('data/jobs')

            # Save to markdown files
            jobs_saved = 0
            for _, job in combined_jobs.iterrows():
                job_id = job['id']
                markdown_content = f"# {job['title']}\n\n"
                markdown_content += f"**Company:** {job['company']}\n\n"
                markdown_content += f"**Location:** {job['location']}\n\n"
                markdown_content += f"**Job Type:** {job['job_type']}\n\n"
                markdown_content += f"**Date Posted:** {job['date_posted']}\n\n"
                markdown_content += f"**Salary:** {job['min_amount']} - {job['max_amount']} {job['currency']} ({job['interval']})\n\n"
                markdown_content += f"**Remote:** {job['is_remote']}\n\n"
                markdown_content += f"**Description:**\n\n{job['description']}\n\n"
                markdown_content += f"**Job URL:** {job['job_url']}\n\n"

                with open(f"data/jobs/{job_id}.md", "w", encoding="utf-8") as file:
                    file.write(markdown_content)
                jobs_saved += 1

            return f"Successfully scraped and saved {jobs_saved} job listings to data/jobs/"

        except Exception as e:
            return f"Error during job scraping: {str(e)}"

    def act(self, query: str) -> tuple:
        """
        Process a natural language query to perform job search and answer using RAG.
        
        Args:
            query (str): Natural language query describing the job search
            
        Returns:
            tuple: (answer, docs, sources) where answer is the generated response,
                  docs are the relevant documents, and sources are the source files
        """
        try:
            # Load job search prompt template
            with open("prompts/job_search_prompt.txt", "r", encoding='utf-8') as f:
                template = f.read()
            
            prompt_template = PromptTemplate.from_template(template)
            formatted_prompt = prompt_template.format(query=query)

            # Get job search parameters through LLM
            content, tool_calls = LLM_answer_v3(
                prompt=formatted_prompt,
                model_name=self.model_name,
                llm_provider=self.llm_provider,
                tool_list=[JobAgent.scrape_and_convert],
                stream=False
            )

            print(f"LLM Response: {content}")
            print(f"Tool calls: {tool_calls}")
            
            # Execute the scraping
            if tool_calls:
                tool_call = tool_calls[0]  # We expect only one tool call
                result = JobAgent.scrape_and_convert.invoke(tool_call['args'])
                print(f"Scraping result: {result}")
                
                # Initialize RAG agent for answering
                from src.main_utils.generation_utils_v2 import RAGAgent
                
                self.config["data_sources"] = ["jobs"]
                self.config["enable_source_filter"] = True
                self.config["field_filter"] = ["jobs"]
                
                rag_agent = RAGAgent(default_config=self.config, config={"stream": True}, qdrant_client=self.qdrant_client)
                answer, docs, sources = rag_agent.RAG_answer(query)
                return answer, docs, sources

            return None, [], []

        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            print(f"Error processing job search query: {error_message}")
            return None, [], []

if __name__ == "__main__":
    # Example usage
    agent = JobAgent(
        model_name="gemini-2.0-flash",
        llm_provider="google",
        is_remote=False
    )
    
    # Test the act method with a natural language query
    query = "Find me data science jobs in Marseille and Aix-en-Provence"
    success = agent.act(query)
    print(f"Job search operation {'succeeded' if success else 'failed'}")