import asyncio
import os

import google.generativeai as genai
import yaml
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from dotenv import load_dotenv
from fake_useragent import UserAgent
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

load_dotenv()


class InternetAgent:
    def __init__(self):
        self.list_pages = []
        # load the internet/internet_config.yaml from the config folder
        with open("config/internet_config.yaml", "r") as file:
            self.internet_config = yaml.safe_load(file)
            
        # self.js_code=[  # Pass js_code via CrawlerRunConfig
        #             """
        #                 console.log("Running cookie consent Javascript...");

        #                 // Try to click the accept button using the ID selector
        #                 const acceptButton = document.querySelector('#axeptio_btn_acceptAll');
        #                 if (acceptButton) {
        #                     console.log("Found accept button by ID '#axeptio_btn_acceptAll', clicking...");
        #                     acceptButton.click();
        #                 } else {
        #                     console.log("Accept button with ID '#axeptio_btn_acceptAll' not found.");
        #                 }
        #             """,
        #         ]



    async def fetch_content(self, url):
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
        from crawl4ai import AsyncWebCrawler, CacheMode
        from crawl4ai.content_filter_strategy import BM25ContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        try:
            async with AsyncWebCrawler(verbose=False, headless=True) as crawler:
                result = await crawler.arun(
                    url=url,
                    cache_mode=CacheMode.BYPASS,
                )
                return result.markdown_v2.raw_markdown
        except Exception as e:
            print(f"[ERROR] 🚫 Failed to crawl {url}, error: {e}")
            return ""

    def scrape_contents(self, urls):
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

    async def scrape_multiple_contents(self, urls):
        """
        Scrape multiple URLs in parallel using Crawl4AI and return markdown.

        Args:
            urls (list): A list of URLs to scrape.

        Returns:
            list: A list of scraped contents in markdown format.
        """
        #fake a user agent each time to avoid detection
        ua = UserAgent()
        #generate random user agent
        random_profile=ua.random
        
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True, #dont forget to put to True after debugging
            verbose=True,
            java_script_enabled=True,
            text_mode=True,  # return the text content of the page without the images and css
            #ça a l'air de débugger Welcome to the jungle !!!
            
            # headers={
            #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            #     "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7", # set the language to French
            # },
            # cookies=[ #bypass the cookie consent
            #     {
            #         "name": "axeptio_cookies",
            #         "value": "true",
            #         "domain": ".welcometothejungle.com",
            #         "path": "/",
            #     }
            # ],
            user_agent_mode=None,  # randomize the user agent to avoid detection with "random"
            user_agent=random_profile,  # set the user agent to a random profile
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=DefaultMarkdownGenerator(),
                wait_until='networkidle', #Very usefull to wait for the page to be fully loaded
                delay_before_return_html=5.0, 
                #js_code=self.js_code, #WASNT THERE BEFORE, JUST TO BY PASS COOKIES
            )
            results = await asyncio.gather(
                *[crawler.arun(url, config=run_config) for url in urls]
            )

        # print("######################")
        # print("RESULTS: ", results)
        # print("######################")

        contents = []
        for result in results:
            if result.error_message:
                print(
                    f"[ERROR] 🚫 Failed to crawl {result.url}, error: {result.error_message}"
                )
                contents.append("")
            elif hasattr(result, "markdown") and result.markdown:
                contents.append(result.markdown)
            else:
                print(f"[ERROR] 🚫 Failed to crawl {result.url}, no markdown available")
                contents.append("")
        return contents

    def scrap_multiple_urls(self, urls, max_urls=None):
        """Function that fetches all the content of the urls gievn but in a parallel way, using the async function scrape_multiple_contents
        we truncate the list of urls to max_urls if max_urls is not None to avoid overcharging the system.

        Args:
            urls (list): list of urls to fetch content from

        Returns:
            list: list of contents of the urls

        """

        if max_urls:
            urls = urls[:max_urls]

        return asyncio.run(self.scrape_multiple_contents(urls))

    def save_pages(self):
        """
        Save the pages in a file giving name page_1, page_2, etc. save them in data/internet folder
        """
        for i, content in enumerate(self.list_pages):
            with open(f"data/internet/page_{i + 1}.md", "w", encoding="utf-8") as f:
                f.write(content)

    def get_urls(self, query, num_results):
        from googlesearch import search

        urls = []
        try:
            for url in search(
                query,
                num_results=num_results,
                lang="fr",
            ):
                urls.append(url)
        except Exception as e:
            print(f"Erreur lors de la recherche : {str(e)}")
        print(f"URLs trouvées : {len(urls)}")
        return urls

    def get_urls_v2(self, query: str, num_results: int = 5) -> list:
        """
        Effectue une recherche avec l'API Google Programmable Search.

        Args:
            query (str): La requête de recherche.
            api_key (str): Votre clé d'API Google Cloud.
            cse_id (str): L'identifiant de votre moteur de recherche programmable.
            num_results (int): Le nombre de résultats souhaités.

        Returns:
            list: Une liste d'URL de résultats de recherche.
        """
        from json import load

        from googleapiclient.discovery import build

        service = build(
            "customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_API_KEY")
        )
        results = (
            service.cse()
            .list(q=query, cx=os.getenv("SEARCH_ENGINE_ID"), num=num_results)
            .execute()
        )

        urls = []
        if "items" in results:
            for item in results["items"]:
                urls.append(item["link"])
        return urls

    def fill_internet_vectorstore(self):
        from src.main_utils.vectorstore_utils_v4 import VectorAgent

        # create a vector agent
        vector_agent = VectorAgent(default_config=self.internet_config)
        # fill the vector store with the pages
        vector_agent.fill()

    def search(self, query, num_urls):
        """get the urls and scrape the contents
        Args:
            query (str): The search query string.
            num_urls (int): The number of search results to retrieve.
        Returns:
            list: A list of contents from the provided URLs. If a URL's content
            is None, it will be replaced with an empty string.
        """

        # get urls
        urls = self.get_urls_v2(query, num_results=num_urls)
        # scrape contents
        contents = self.scrape_contents(urls)
        return contents

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
        cleaned_content = re.sub(
            r"http\S+|www\S+|https\S+", "", raw_content, flags=re.MULTILINE
        )
        # Remove session redirects and tracking parameters
        cleaned_content = re.sub(
            r"\b(?:trk|session_redirect|ProductId)\S*", "", cleaned_content
        )
        # Remove special characters and extra whitespace
        cleaned_content = re.sub(r"\s+", " ", cleaned_content)
        cleaned_content = re.sub(r'[^A-Za-z0-9À-ÿ\s.,!?\'"-]', "", cleaned_content)
        # Remove LinkedIn-specific phrases
        linkedin_phrases = [
            "See more",
            "Play Video",
            "Video Player is loading",
            "Loaded",
            "PlayBack to start",
            "Stream Type LIVE",
            "Current Time",
            "Duration",
            "Playback Rate",
            "Show Captions",
            "Mute",
            "Fullscreen",
            "Like",
            "Comment",
            "Share",
            "Report this post",
            "Report this comment",
            "Reply",
            "Reactions",
            "Followers",
            "Posts",
            "View Profile",
            "Follow",
            "Explore topics",
            "Sign in",
            "Join now",
            "Continue with Google",
            "Welcome back",
            "Email or phone",
            "Password",
            "Forgot password",
        ]
        pattern = "|".join(map(re.escape, linkedin_phrases))
        cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.IGNORECASE)
        # Remove extra spaces again after removals
        cleaned_content = re.sub(r"\s{2,}", " ", cleaned_content).strip()
        return cleaned_content


class GeminiInternetAgent:
    """
    A class to interact with the Gemini API for generating content with internet-grounded search.
    Attributes:
        client (genai.Client): The client to interact with the Gemini API.
        model_id (str): The ID of the model to use for generating content.
        google_search_tool (Tool): The tool for performing Google searches.
    Methods:
        __init__():
            Initializes the GeminiInternetAgent with the necessary API key and tools.
        answer(prompt: str) -> str:
                prompt (str): The question or prompt to answer.
                str: A string containing the answer.
        answer_and_get_grounding_metadata(prompt: str) -> tuple[str, str]:
                prompt (str): The question or prompt to answer.
                tuple[str, str]: A tuple containing the answer and the grounding metadata.
    """

    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variable")

        self.client = genai.Client(api_key=api_key)  # API key passed to client
        self.model_id = "gemini-2.0-flash-exp"  # Make sure this model id is correct
        self.google_search_tool = Tool(google_search=GoogleSearch())

    def answer(self, prompt: str) -> str:
        """
        Answers a prompt using Gemini with internet-grounded search.

        Args:
            prompt: The question or prompt to answer.

        Returns:
            A string containing the answer.
        """
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[self.google_search_tool],
                response_modalities=["TEXT"],
            ),
        )
        if response.candidates and response.candidates[0].content.parts:
            answer = "".join(part.text for part in response.candidates[0].content.parts)
        else:
            answer = "No answer available."

        return answer

    def answer_and_get_grounding_metadata(self, prompt: str) -> tuple[str, str]:
        """
         Answers a prompt using Gemini with internet-grounded search and returns grounding metadata.

        Args:
            prompt: The question or prompt to answer.

        Returns:
            A tuple containing the answer and the grounding metadata.
        """
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[self.google_search_tool],
                response_modalities=["TEXT"],
            ),
        )
        if response.candidates and response.candidates[0].content.parts:
            answer = "".join(part.text for part in response.candidates[0].content.parts)
        else:
            answer = "No answer available."

        grounding_metadata = (
            response.candidates[
                0
            ].grounding_metadata.search_entry_point.rendered_content
            if response.candidates
            and response.candidates[0].grounding_metadata
            and hasattr(response.candidates[0].grounding_metadata, "search_entry_point")
            and hasattr(
                response.candidates[0].grounding_metadata.search_entry_point,
                "rendered_content",
            )
            else "No grounding metadata available."
        )
        return answer, grounding_metadata


if __name__ == "__main__":
    # define user's query

    urls = ["https://www.lemonde.fr/", "https://www.lefigaro.fr/"]

    # scrap multiple urls
    agent = InternetAgent()
    contents = agent.scrap_multiple_urls(urls)
    print("Contents: ", contents)

    # test scrapp multiple contents
    # async def main():
    #     agent = InternetAgent()
    #     urls = ['https://www.lemonde.fr/', 'https://www.lefigaro.fr/']
    #     contents = await agent.scrape_multiple_contents(urls)
    #     #print("Contents: ", contents)
    #     return contents

    # res=asyncio.run(main())
    # print("Contents: ", res)

    # q = "Qui est Pierre-Yves Rougeyron ?"
    # #get urls
    # urls = agent.get_urls_v2(query=q, num_results=5)
    # print("Urls: ", urls)

    # #scrape contents
    # urls=['https://www.linkedin.com/posts/salom%C3%A9-saqu%C3%A9-499563105_cest-la-campagne-%C3%A9lectorale-la-plus-violente-activity-7256560321585205248-Te2A?utm_source=share&utm_medium=member_desktop']

    # contents = agent.scrape_contents(urls)
    # #print contents separately, correctly presented
    # # for i, content in enumerate(contents):
    # #     print(f"Content {i+1}: {content}\n")
    # print("Contents: ", contents)

    # clean the content
    # cleaned_content = agent.linkedin_post_cleaner(contents[0])
    # print(f"Cleaned content: {cleaned_content}")
    # Example usage:
    # agent = GeminiInternetAgent()
    # question = "What are today news and today date ?"
    # answer = agent.answer(question)
    # print(f"Question: {question}")
    # print(f"Answer: {answer}")
