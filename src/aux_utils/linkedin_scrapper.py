from linkedin_scraper import JobSearch, actions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up Chrome options
chrome_options = Options()
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--headless")

# Initialize the WebDriver
driver = webdriver.Chrome(options=chrome_options)

# Load email and password from .env file
email = os.getenv("EMAIL")
password = os.getenv("PASSWORD")

# Log in to LinkedIn
actions.login(driver, email, password)  # if email and password isn't given, it'll prompt in terminal

# Wait for user input to proceed
input("Press Enter to continue...")

# Initialize JobSearch
job_search = JobSearch(driver=driver, close_on_complete=False, scrape=False)

# Search for jobs
job_listings = job_search.search("Machine Learning Engineer")  # returns the list of `Job` from the first page

# Print job listings
for job in job_listings:
    print(job)