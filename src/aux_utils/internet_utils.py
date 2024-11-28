import asyncio
import yaml

class InternetAgent:
    
    def __init__(self):
        self.list_pages = []
        #load the internet/internet_config.yaml from the config folder
        with open("config/internet_config.yaml", "r") as file:
            self.internet_config = yaml.safe_load(file)
    
    async def fetch_content(self,url):
        """
        Asynchronously fetches the content of a given URL and returns it in markdown format.

        This function uses the AsyncWebCrawler from the crawl4ai library to perform the web crawling.
        If an error occurs during the crawling process, it catches the exception, logs an error message,
        and returns an empty string.

        Args:
            url (str): The URL of the web page to fetch content from.

        Returns:
            str: The content of the web page in markdown format, or an empty string if an error occurs.
        """
        from crawl4ai import AsyncWebCrawler
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)
                return result.markdown
        except Exception as e:
            print(f"[ERROR] 🚫 Failed to crawl {url}, error: {e}")
            return ""
        
    def scrape_contents(self,urls):
        """
        Scrape the contents of multiple URLs asynchronously.

        Args:
            urls (list): A list of URLs to scrape.

        Returns:
            list: A list of contents from the provided URLs. If a URL's content 
            is None, it will be replaced with an empty string.
        """
        loop = asyncio.get_event_loop()
        tasks = [self.fetch_content(url) for url in urls]
        contents = loop.run_until_complete(asyncio.gather(*tasks))
        # Ensure all contents are strings
        contents = [content if content is not None else "" for content in contents]
        self.list_pages = contents
        return contents
    
    def save_pages(self):
        """
        Save the pages in a file giving name page_1, page_2, etc. save them in data/internet folder
        """
        for i, content in enumerate(self.list_pages):
            with open(f"data/internet/page_{i+1}.md", "w", encoding="utf-8") as f:
                f.write(content)
    
    
    def get_urls(self,query, num_results):
        """
        Retrieve a list of URLs based on a search query.

        This function uses the `googlesearch` module to perform a search
        and returns a list of URLs that match the query.

        Args:
            query (str): The search query string.
            num_results (int): The number of search results to retrieve.

        Returns:
            list: A list of URLs that match the search query.
        """
        from googlesearch import search
        urls = []
        for url in search(query, num_results=num_results,lang='fr'):
            urls.append(url)
        print("Number or urls found by get_url: ", len(urls))
        return urls
    
    def fill_internet_vectorstore(self):
        from src.main_utils.vectorstore_utils_v2 import VectorAgent
        
        #create a vector agent
        vector_agent = VectorAgent(default_config=self.internet_config)
        #fill the vector store with the pages
        vector_agent.fill()
    
    
    def linkedin_post_cleaner(self, raw_content):
        """
        Clean the content of a LinkedIn post by removing unwanted elements.

        Args:
            raw_content (str): The raw content of the LinkedIn post.

        Returns:
            str: The cleaned content of the LinkedIn post.
        """
        import re
        # Remove URLs
        cleaned_content = re.sub(r'http\S+|www\S+|https\S+', '', raw_content, flags=re.MULTILINE)
        # Remove session redirects and tracking parameters
        cleaned_content = re.sub(r'\b(?:trk|session_redirect|ProductId)\S*', '', cleaned_content)
        # Remove special characters and extra whitespace
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        cleaned_content = re.sub(r'[^A-Za-z0-9À-ÿ\s.,!?\'"-]', '', cleaned_content)
        # Remove LinkedIn-specific phrases
        linkedin_phrases = [
            "See more", "Play Video", "Video Player is loading", "Loaded",
            "PlayBack to start", "Stream Type LIVE", "Current Time",
            "Duration", "Playback Rate", "Show Captions", "Mute", "Fullscreen",
            "Like", "Comment", "Share", "Report this post", "Report this comment",
            "Reply", "Reactions", "Followers", "Posts", "View Profile", "Follow",
            "Explore topics", "Sign in", "Join now", "Continue with Google",
            "Welcome back", "Email or phone", "Password", "Forgot password"
        ]
        pattern = '|'.join(map(re.escape, linkedin_phrases))
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE)
        # Remove extra spaces again after removals
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content).strip()
        return cleaned_content



if __name__ == "__main__":
    agent = InternetAgent()
    #define user's query
    # query = "Qui est Pierre-Yves Rougeyron ?"
    # #get urls
    # urls = agent.get_urls(query, num_results=5)
    #scrape contents
    urls=['https://www.linkedin.com/posts/salom%C3%A9-saqu%C3%A9-499563105_cest-la-campagne-%C3%A9lectorale-la-plus-violente-activity-7256560321585205248-Te2A?utm_source=share&utm_medium=member_desktop']
    
    contents = agent.scrape_contents(urls)
    #print contents separately, correctly presented
    # for i, content in enumerate(contents):
    #     print(f"Content {i+1}: {content}\n")
    print("Contents: ", contents)
    
    #clean the content
    #cleaned_content = agent.linkedin_post_cleaner(contents[0])
    #print(f"Cleaned content: {cleaned_content}")
            
    
        





