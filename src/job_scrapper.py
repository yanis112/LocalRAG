import os
from jobspy import scrape_jobs

class JobScrapper:
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
    def __init__(self, search_term, location, hours_old, results_wanted):
        self.search_term = search_term
        self.location = location
        self.hours_old = hours_old
        self.results_wanted = results_wanted

    def scrape_and_convert(self):
        jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "glassdoor"],
            search_term=self.search_term,
            location=self.location,
            results_wanted=self.results_wanted,
            hours_old=self.hours_old,
            country_indeed='france'
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
    scraper = JobScrapper(search_term="Data Scientist", location="Marseille, France", hours_old=600, results_wanted=20)
    scraper.scrape_and_convert()
