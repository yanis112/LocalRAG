# The goal of this file is to manage the links given by the user to extract the associate rescource and save it in the database
import asyncio
import re
import traceback
from typing import Optional

from emoji import config
import streamlit as st
from crawl4ai import AsyncWebCrawler, CacheMode

from src.aux_utils.logging_utils import setup_logger

link_logger = setup_logger(__name__, "external_resource_scrapper.log")


def clean_linkedin_post(text):
    link_logger.info("Cleaning LinkedIn post content.")
    # Remove cookie/consent banner and header UI
    text = re.sub(
        r".*?(and 3rd parties use.*?Cookie Policy\..*?Join now Sign in)",
        "",
        text,
        flags=re.DOTALL,
    )
    link_logger.debug("Removed cookie/consent banner and header UI.")

    # Remove LinkedIn header/cookie notice
    text = re.sub(
        r"LinkedIn and 3rd parties.*?Cookie Policy\..*?Sign in",
        "",
        text,
        flags=re.DOTALL,
    )
    link_logger.debug("Removed LinkedIn header/cookie notice.")

    # Remove everything before the post content
    text = re.sub(r".*Post\n", "", text, flags=re.DOTALL)
    link_logger.debug("Removed content before the post.")

    # Remove profile information and timestamps
    # text = re.sub(r'\n.*?\d+[mdh]\s*(Edited)?\s*', '', text)

    # Remove empty links and references
    text = re.sub(r"\[\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\](\([^)]+\))?", "", text)
    link_logger.debug("Removed empty links and references.")

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    link_logger.debug("Removed URLs.")

    # Remove social interaction elements and counts
    text = re.sub(r"\d+\s*(Reactions?|Comments?|shares?)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(Like|Comment|Share|Copy|Report this)\s*(post|comment)?", "", text)
    text = re.sub(r"(LinkedIn|Facebook|Twitter)", "", text)
    text = re.sub(r"To view or add a comment, sign in.*", "", text, flags=re.DOTALL)
    link_logger.debug("Removed social interaction elements and counts.")
    # Remove hashtags and "More Relevant Posts" section
    text = re.sub(r"(#\w+\s*)+", "", text)
    text = re.sub(r"More Relevant Posts.*", "", text, flags=re.DOTALL)
    link_logger.debug("Removed hashtags and 'More Relevant Posts' section.")

    link_logger.info("Finished cleaning LinkedIn post content.")
    return text.strip()


def clean_scrapped_content(
    text: str, model_name: str, llm_provider: str, resource_type: str = "linkedin"
) -> str:
    """
    Takes a text crawled from a website and cleans it by removing irrelevant noisy infos using an LLM model.

    Args:
        text (str): The text to be cleaned.
        model_name (str): The name of the LLM model to use for cleaning.
        llm_provider (str): The provider of the LLM model.

    Returns:
        str: The cleaned text.

    """
    link_logger.info(f"Cleaning scrapped content. Resource type: {resource_type}")
    from langchain_core.prompts import PromptTemplate
    from src.main_utils.generation_utils_v2 import LLM_answer_v3

    if resource_type == "linkedin":
        cleaning_prompt_path = "prompts/linkedin_cleaning.txt"
    else:
        cleaning_prompt_path = "prompts/website_cleaning.txt"
    link_logger.debug(f"Using cleaning prompt from: {cleaning_prompt_path}")

    # load the cleaning prompt template from a txt file
    with open(cleaning_prompt_path, "r", encoding="utf-8") as file:
        cleaning_prompt = file.read()

        # Instantiation using from_template (recommended)
        cleaning_prompt = PromptTemplate.from_template(cleaning_prompt)
        # format the prompt
        full_prompt = cleaning_prompt.format(resource=text)
    link_logger.debug(f"Formatted prompt: {full_prompt[:500]}...")  # Log only a part for brevity

    cleaned_text = LLM_answer_v3(
        prompt=full_prompt,
        model_name=model_name,
        llm_provider=llm_provider,
        stream=False,
    )
    link_logger.info("Content cleaned successfully using LLM.")
    return cleaned_text


def extract_resource(
    link: str, timeout: int = 60, resource_type: str = "linkedin"
) -> Optional[str]:
    """
    Takes a Linkedin post link as input and extracts the content of the post.

    Args:
        link (str): The URL of the Linkedin post.
        timeout (int): Maximum time in seconds to wait for extraction.

    Returns:
        Optional[str]: The extracted content of the post, or None if extraction fails.
    """
    link_logger.info(f"Extracting resource from link: {link}")
    # Input validation
    if not link or not isinstance(link, str):
        link_logger.error("Link must be a non-empty string")
        raise ValueError("Link must be a non-empty string")

    # Clean link
    link = link.strip()
    link_logger.debug(f"Cleaned link: {link}")

    # Set Windows event loop policy
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    link_logger.debug("Set Windows event loop policy.")

    async def main(url: str) -> Optional[str]:
        try:
            from fake_useragent import UserAgent
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

            ua = UserAgent()
            # generate random user agent
            random_profile = ua.random

            crawler_config = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(),
                cache_mode=CacheMode.BYPASS,
                wait_for="document.readyState === 'complete'", #TRES UTILE POUR WELCOME TO THE JUNGLE
                page_timeout=30000, # 3 seconds timeout for page load  #TRES UTILE POUR WELCOME TO THE JUNGLE
                 # Petite pause avant de capturer le HTML
                delay_before_return_html=2.0, #TRES UTILE
                #wait_for="networkidle",
                # Injecte le JS qui clique sur le bouton de cookies
                js_code=[
                    """
    document.addEventListener('DOMContentLoaded', function() {
        var acceptButton = document.querySelector('#axeptio_btn_acceptAll');
        if (acceptButton) {
            acceptButton.click();
        }
    });
    """,
                    "document.querySelector('#axeptio_btn_acceptAll')?.click();",
                ],
            )

            browser_config = BrowserConfig(
                browser_type="chromium",
                headless=False,  # dont forget to put to True after debugging
                verbose=True,
                java_script_enabled=True,
                text_mode=False,  # was True before  # return the text content of the page without the images and css
                user_agent_mode=None,  # randomize the user agent to avoid detection with "random"
                user_agent=random_profile,  # set the user agent to a random profile
            )
            async with AsyncWebCrawler(
                verbose=True, config=browser_config
            ) as crawler:
                # Add timeout to the crawl operation
                link_logger.debug(f"Crawling URL: {url} with timeout {timeout} seconds.")
                raw_html_result = await asyncio.wait_for(
                    crawler.arun(
                        url=url, cache_mode=CacheMode.BYPASS, config=crawler_config
                    ),  # the config parameter was added ! not there before
                    timeout=timeout,
                )
                link_logger.debug("Raw HTML result obtained.")
                print("RAW HTML RESULT:", raw_html_result)
                print("####################")
                raw_markdown_result = (
                    raw_html_result.markdown
                )  # markdown_v2.raw_markdown #) AVANT MAIS NE MARCHE PAS EST VIDE
                link_logger.info(f"Raw Markdown scrapped:\n{raw_markdown_result}")  # log the raw markdown scrapped

                if not raw_markdown_result:
                    link_logger.warning("No content extracted from the link !")
                    return None

                print("####################")
                print("RAW MARKDOWN RESULT:", raw_markdown_result)
                print("####################")
                cleaned_markdown_result = clean_scrapped_content(
                    raw_markdown_result,
                    model_name="llama-3.3-70b-versatile",
                    llm_provider="groq",
                    resource_type=resource_type,
                )  # clean the resource depending on its type (will change the cleaning prompt depending on if it's from a website or linkedin)

                link_logger.info(f"Cleaned text:\n{cleaned_markdown_result}")  # log the cleaned markdown result
                return cleaned_markdown_result # Return the cleaned markdown result

        except asyncio.TimeoutError:
            link_logger.warning(f"Extraction timed out after {timeout} seconds")
            return None
        except Exception as e:
            link_logger.error(f"Error extracting content: {str(e)}")
            traceback.print_exc()
            return None

    try:
        # Run the async function with new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        link_logger.debug("Running async extraction.")
        content = loop.run_until_complete(main(link))
        loop.close()

        return content

    except Exception as e:
        link_logger.error(f"Error in event loop: {str(e)}")
        traceback.print_exc()
        return None


class ExternalKnowledgeManager:
    def __init__(self, config, client=None):
        link_logger.info("Initializing ExternalKnowledgeManager.")
        self.current_transcription = None
        self.config = config
        self.client = client

    def classify_rescource(self, link):
        """
        Classify the given link as a specific type of resource.
        This method checks the provided link and classifies it as either a
        YouTube video, a LinkedIn profile, or a general website based on
        the content of the link.
        Args:
            link (str): The URL or link to be classified.
        Returns:
            str: The type of resource the link represents. Possible values
            are 'video' for YouTube videos, 'linkedin' for LinkedIn profiles,
            and 'website' for general websites.
        """
        link_logger.info(f"Classifying resource: {link}")

        if "youtu" in str(link):
            link_logger.info("This is a youtube video !")
            return "video"
        elif "linkedin" in str(link):
            link_logger.info("This is a linkedin post !")
            return "linkedin"
        else:
            link_logger.info("This is a website !")
            return "website"

    def extract_rescource_from_link(self, link):
        """
        Extracts the resource from the given link based on its classification.
        This method classifies the provided link and extracts the resource accordingly.
        If the link is classified as a video, it uses YouTubeTranscriber to transcribe the video.
        If the link is classified as a LinkedIn resource, it extracts the LinkedIn resource.
        Otherwise, it extracts the resource from a website.
        Args:
            link (str): The URL of the resource to be extracted.
        Returns:
            str: The extracted transcription if the resource is a video, or the extracted content from LinkedIn or a website.
        """
        link_logger.info(f"Extracting resource from link: {link}")

        with st.spinner("⛏️ Extracting the url content ...", show_time=True):
            if self.classify_rescource(link) == "video":
                st.toast("Video resource detected!", icon="📺")
                from src.aux_utils.transcription_utils_v3 import YouTubeTranscriber

                transcriber = YouTubeTranscriber(
                    chunk_size=st.session_state["config"]["transcription_chunk_size"], batch_size=1
                )
                # YouTubeTranscriber(chunk_size=self.config["transcription_chunk_size"],batch_size=1)
                # transcription = transcriber.transcribe(
                #     input_path=link, method="groq",diarization=True
                # )
                transcription = transcriber.transcribe(
                    input_path=link,
                    method="groq",
                    diarization=st.session_state["diarization_enabled"],
                )

                self.current_rescource = "### This is a video transcription ### " + transcription
                link_logger.info("Video resource extracted successfully !")
                link_logger.debug(f"Extracted transcription: {transcription[:500]}...")
                return transcription

            elif self.classify_rescource(link) == "linkedin":
                rescource = str(extract_resource(link, resource_type="linkedin"))
                self.current_rescource = "### This is a linkedin post ### " + rescource
                link_logger.info("Linkedin resource extracted successfully !")
                return rescource
            else:
                # extract using website type
                rescource = str(extract_resource(link, resource_type="website"))
                self.current_rescource = "### This is a website content ### " + rescource
                link_logger.info("Website resource extracted successfully !")
                return rescource

    def extract_rescource_from_raw_text(self, text):
        """Extract the rescource from the given text. no link.
        Args:
            text (str): The text to be extracted.
        """
        link_logger.info("Extracting resource from text.")
        self.current_rescource = text.replace("```markdown", "").replace(
            "```", ""
        )  # remove markdown code block indicators
        link_logger.info("Resource extracted successfully !")
        link_logger.debug(f"Extracted resource: {text[:500]}...")
        return text

    def index_rescource(self):
        """
        Indexes the current transcription as a resource.
        This method performs the following steps:
        1. Classifies the current transcription to determine its topic.
        2. Saves the transcription in a Markdown file in the appropriate directory based on the topic.
        3. Indexes the saved resource in the database.
        The directory for saving the resource is determined by the topic classification,
        which maps to a directory specified in the configuration.
        Imports:
            time: Standard library module for time-related functions.
            IntentClassifier: Class for classifying text into predefined topics.
            VectorAgent: Class for handling vector-based operations in the database.
        Raises:
            Any exceptions raised during file operations or classification will propagate up.
        Side Effects:
            Creates a new Markdown file in the appropriate directory.
            Prints a success message upon saving the resource.
        """
        link_logger.info("Indexing resource.")
        import time

        # First we save the rescource in a file in the appropriate folder
        from src.aux_utils.text_classification_utils import IntentClassifier

        with st.spinner("Indexing the resource...", show_time=True):
            topic_classifier = IntentClassifier(
                config=self.config, labels_dict=self.config["data_sources"]
            )

            # get the associated topic description
            topic = topic_classifier.classify(self.current_rescource, method="LLM")

            link_logger.info(f"Topic found by classifier: {topic}")
            st.toast(f"Associated Topic: {topic}")

            # get the directory associated to the topic (the key associated to the topic in the dictionary)
            directory = topic
            # save the rescource in the data/{directory} folder in .md format
            file_path = f"data/{directory}/resource_{time.time()}.md"
            with open(
                file_path,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(self.current_rescource)
                link_logger.info(f"Resource saved successfully in {file_path}")

            # we index the rescource in the database !
            from src.main_utils.vectorstore_utils_v5 import VectorAgent
            # check if qdrant client is in session state

            vector_agent = VectorAgent(default_config=self.config, qdrant_client=self.client)
            vector_agent.build_vectorstore()
            link_logger.info("Resource indexed in the database.")


if __name__ == "__main__":
    # load config using yaml from config/config.yaml
    import yaml
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    manager=ExternalKnowledgeManager(config)
    link="https://www.welcometothejungle.com/fr/companies/inoco/jobs/developpeur-ia-f-h_paris?q=61f7ded5e9ee049906b00987edc59132&o=266b2d6c-2211-4390-b5a8-4686576d11ca"
    manager.extract_rescource_from_link(link)
    # manager.index_rescource()
    #link = "https://www.welcometothejungle.com/fr/companies/inoco/jobs/developpeur-ia-f-h_paris?q=61f7ded5e9ee049906b00987edc59132&o=266b2d6c-2211-4390-b5a8-4686576d11ca"
  