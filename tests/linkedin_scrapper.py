from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup
from time import sleep

class LinkedinPostScrapper:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    def fetch_content(self, url):
        try:
            self.driver.get(url)
            sleep(3)  # Wait for the page to load
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            post_content = soup.find('div', {'class': 'feed-shared-update-v2__description'}).get_text(separator="\n")
            return post_content
        except Exception as e:
            print(f"[ERROR] 🚫 Failed to scrape {url}, error: {e}")
            return ""
    
    def scrape_contents(self, urls):
        contents = [self.fetch_content(url) for url in urls]
        return contents

    def save_pages(self, contents):
        for i, content in enumerate(contents):
            with open(f"data/internet/page_{i+1}.md", "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    scrapper = LinkedinPostScrapper()
    urls = [
        "https://www.linkedin.com/posts/salom%C3%A9-saqu%C3%A9-499563105_cest-la-campagne-%C3%A9lectorale-la-plus-violente-activity-7256560321585205248-Te2A?utm_source=share&utm_medium=member_desktop",
    ]
    contents = scrapper.scrape_contents(urls)
    print(f"Contents: {contents}")
    # for i, content in enumerate(contents):
    #     print(f"Content {i+1}:\n{content}\n")
    # scrapper.save_pages(contents)