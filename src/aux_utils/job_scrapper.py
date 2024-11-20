import os
from jobspy import scrape_jobs

class JobAgent:
    """
    A class used to scrape job listings from various websites and convert them to markdown files.
    Attributes
    ----------
    search_term : str
        The term to search for in job listings.
    location : str
        The location to search for job listings.
    hours_old : int
        The maximum age of job listings in hours.
    results_wanted : int
        The number of job listings to retrieve.
    Methods
    -------
    scrape_and_convert():
        Scrapes job listings and converts them to markdown files.
    """
    def __init__(self, search_term, location, hours_old, results_wanted,google_search_term,is_remote=False):
        self.search_term = search_term
        self.location = location
        self.hours_old = hours_old
        self.results_wanted = results_wanted
        self.google_search_term = google_search_term
        self.is_remote = is_remote

    def scrape_and_convert(self):
        """
        Scrape job listings from multiple job sites and convert them to markdown files.
        This method scrapes job listings from Indeed, LinkedIn, and Glassdoor based on the 
        search term, location, and other parameters specified in the instance. It then 
        converts each job listing into a markdown file and saves it in the 'data/jobs' directory.
        The markdown file contains the job title, company, location, job type, date posted, 
        salary range, remote status, job description, and job URL.
        If the 'data/jobs' directory does not exist, it will be created.
        Attributes:
            site_name (list): List of job sites to scrape from.
            search_term (str): The search term for the job listings.
            location (str): The location for the job listings.
            results_wanted (int): The number of job listings to retrieve.
            hours_old (int): The maximum age of job listings in hours.
            country_indeed (str): The country for Indeed job listings.
            is_remote (bool): Whether to filter for remote jobs.
            google_search_term (str): The search term for Google job listings.
        Raises:
            OSError: If there is an issue creating the 'data/jobs' directory or writing the markdown files.
        """
        jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "glassdoor","google"],
            google_search_term=self.google_search_term,
            search_term=self.search_term,
            location=self.location,
            results_wanted=self.results_wanted,
            hours_old=self.hours_old,
            country_indeed='france',
            is_remote=self.is_remote
        )
        print(f"Found {len(jobs)} jobs")
        print(jobs.columns)

        if not os.path.exists('data/jobs'):
            os.makedirs('data/jobs')

        for _, job in jobs.iterrows():
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
    # Example usage
    scraper = JobAgent(search_term="Data Scientist", location="Aix en Provence", hours_old=200, results_wanted=20,google_search_term="Data Scientist Aix en Provence",is_remote=False)
    scraper.scrape_and_convert()
