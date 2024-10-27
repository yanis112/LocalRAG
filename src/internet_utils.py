import asyncio





class InternetAgent:
    
    def __init__(self):
        pass
    
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
        return contents
    
    
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
        return urls

    

if __name__ == "__main__":
    agent = InternetAgent()
    #define user's query
    query = "Qui est Pierre-Yves Rougeyron ?"
    #get urls
    urls = agent.get_urls(query, num_results=5)
    #scrape contents
    contents = agent.scrape_contents(urls)
    #print contents separately, correctly presented
    for i, content in enumerate(contents):
        print(f"Content {i+1}: {content}\n")
            
    
        





