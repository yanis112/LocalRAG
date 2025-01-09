import os
from functools import lru_cache
import streamlit as st
from time import sleep
import pandas as pd
from langchain_core.tools import tool
    
    
class JobAgent:
    """
    A class used to represent a Job Agent that scrapes job listings from various websites.
    
    Attributes
    ----------
    search_terms : Union[str, List[str]], optional
        Single term or list of terms to search for in job listings (default is None)
    locations : Union[str, List[str]], optional
        Single location or list of locations to search for job listings (default is None)
    hours_old : int, optional
        The maximum age of job listings in hours (default is None)
    results_wanted : int, optional
        Total number of job listings to fetch across all searches (default is None)
    google_search_terms : Union[str, List[str]], optional
        Search terms for Google job listings, should match search_terms (default is None)
    is_remote : bool, optional
        Whether to search for remote jobs (default is False)
    """
    def __init__(self, search_terms=None, locations=None, hours_old=None, 
                 results_wanted=None, google_search_terms=None, is_remote=False):
        self.search_terms = [search_terms] if isinstance(search_terms, str) else search_terms
        self.locations = [locations] if isinstance(locations, str) else locations
        self.hours_old = hours_old
        self.results_wanted = results_wanted
        self.google_search_terms = [google_search_terms] if isinstance(google_search_terms, str) else google_search_terms
        self.is_remote = is_remote

    @lru_cache(maxsize=None)
    def _fetch_jobs(self, search_term: str, location: str, hours_old: int, 
                   results_wanted: int, google_search_term: str, is_remote: bool):
        """
        Fetch jobs for a single search term and location combination.
        
        Args:
            search_term (str): Single search term
            location (str): Single location
            hours_old (int): Maximum age of listings
            results_wanted (int): Number of results for this specific search
            google_search_term (str): Google-specific search term
            is_remote (bool): Remote job filter
            
        Returns:
            pd.DataFrame: Job listings or None if error occurs
        """
        try:
            from jobspy import scrape_jobs
            return scrape_jobs(
                site_name=["indeed", "linkedin", "glassdoor", "google"],
                google_search_term=google_search_term,
                search_term=search_term,
                location=location,
                results_wanted=results_wanted,
                hours_old=hours_old,
                country_indeed='france',
                is_remote=is_remote
            )
        except Exception as e:
            st.error(f"Error fetching jobs for {search_term} in {location}: {str(e)}")
            return None

    def scrape_and_convert(self):
        """
        Scrape job listings for all search term and location combinations.
        
        Distributes results_wanted across all search combinations and combines results.
        Converts results to markdown files in the 'data/jobs' directory.
        
        Raises:
            ValueError: If required parameters are not set
        """
        if not all([self.search_terms, self.locations, self.hours_old, 
                   self.results_wanted, self.google_search_terms]):
            raise ValueError("All parameters must be set before scraping")
            
        # Calculate results per combination
        total_combinations = len(self.search_terms) * len(self.locations)
        results_per_search = max(1, self.results_wanted // total_combinations)
        
        all_jobs = []
        
        # Fetch jobs for each combination
        for search_term, google_term in zip(self.search_terms, self.google_search_terms):
            for location in self.locations:
                jobs = self._fetch_jobs(
                    search_term, 
                    location,
                    self.hours_old,
                    results_per_search,
                    google_term,
                    self.is_remote
                )
                if jobs is not None:
                    all_jobs.append(jobs)
                sleep(1)  # Rate limiting
        
        if not all_jobs:
            return
            
        # Combine all results
        combined_jobs = pd.concat(all_jobs, ignore_index=True)
        
        # Create directory if needed
        if not os.path.exists('data/jobs'):
            os.makedirs('data/jobs')

        # Save to markdown files
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

if __name__ == "__main__":
    # Example usage with multiple search terms and locations
    scraper = JobAgent(
        search_terms=["Data Scientist", "Machine Learning Engineer"],
        locations=["Aix en Provence", "Marseille"],
        hours_old=200,
        results_wanted=40,
        google_search_terms=["Data Scientist Provence", "Data Engineer Provence"],
        is_remote=False
    )
    scraper.scrape_and_convert()