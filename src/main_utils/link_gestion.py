# The goal of this file is to manage the links given by the user to extract the associate rescource and save it in the database
import asyncio
import re
import traceback
from typing import Optional

import streamlit as st
from crawl4ai import AsyncWebCrawler, CacheMode


def clean_linkedin_post(text):
    # Remove cookie/consent banner and header UI
    text = re.sub(
        r".*?(and 3rd parties use.*?Cookie Policy\..*?Join now Sign in)",
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove LinkedIn header/cookie notice
    text = re.sub(
        r"LinkedIn and 3rd parties.*?Cookie Policy\..*?Sign in",
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove everything before the post content
    text = re.sub(r".*Post\n", "", text, flags=re.DOTALL)

    # Remove profile information and timestamps
    # text = re.sub(r'\n.*?\d+[mdh]\s*(Edited)?\s*', '', text)

    # Remove empty links and references
    text = re.sub(r"\[\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\](\([^)]+\))?", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove social interaction elements and counts
    text = re.sub(
        r"\d+\s*(Reactions?|Comments?|shares?)", "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"(Like|Comment|Share|Copy|Report this)\s*(post|comment)?", "", text
    )
    text = re.sub(r"(LinkedIn|Facebook|Twitter)", "", text)
    text = re.sub(
        r"To view or add a comment, sign in.*", "", text, flags=re.DOTALL
    )
    # Remove hashtags and "More Relevant Posts" section
    text = re.sub(r"(#\w+\s*)+", "", text)
    text = re.sub(r"More Relevant Posts.*", "", text, flags=re.DOTALL)

    return text.strip()


def clean_scrapped_content(
    text: str, model_name: str, llm_provider: str
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
    from langchain_core.prompts import PromptTemplate

    from src.main_utils.generation_utils_v2 import LLM_answer_v3

    # load the cleaning prompt template from a txt file
    with open("prompts/linkedin_cleaning.txt", "r", encoding="utf-8") as file:
        cleaning_prompt = file.read()

        # Instantiation using from_template (recommended)
        cleaning_prompt = PromptTemplate.from_template(cleaning_prompt)
        # format the prompt
        full_prompt = cleaning_prompt.format(post=text)

    return LLM_answer_v3(
        prompt=full_prompt,
        model_name=model_name,
        llm_provider=llm_provider,
        stream=False,
    )


def extract_linkedin(link: str, timeout: int = 30) -> Optional[str]:
    """
    Takes a Linkedin post link as input and extracts the content of the post.

    Args:
        link (str): The URL of the Linkedin post.
        timeout (int): Maximum time in seconds to wait for extraction.

    Returns:
        Optional[str]: The extracted content of the post, or None if extraction fails.
    """
    # Input validation
    if not link or not isinstance(link, str):
        raise ValueError("Link must be a non-empty string")

    # Clean link
    link = link.strip()

    # Set Windows event loop policy
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    async def main(url: str) -> Optional[str]:
        try:
            async with AsyncWebCrawler(verbose=True, headless=True) as crawler:
                # Add timeout to the crawl operation
                result = await asyncio.wait_for(
                    crawler.arun(url=url, cache_mode=CacheMode.BYPASS),
                    timeout=timeout,
                )
                return (
                    clean_scrapped_content(
                        str(result.markdown_v2.raw_markdown),
                        model_name="llama-3.1-8b-instant",
                        llm_provider="groq",
                    )
                    if result and result.markdown
                    else None
                )

        except asyncio.TimeoutError:
            print(f"Extraction timed out after {timeout} seconds")
            return None
        except Exception as e:
            print(f"Error extracting content: {str(e)}")
            traceback.print_exc()
            return None

    try:
        # Run the async function with new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        content = loop.run_until_complete(main(link))
        loop.close()

        return content

    except Exception as e:
        print(f"Error in event loop: {str(e)}")
        traceback.print_exc()
        return None


class ExternalKnowledgeManager:
    def __init__(self, config, client=None):
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

        if "youtu" in str(link):
            print("This is a youtube video !")
            return "video"
        elif "linkedin" in str(link):
            print("This is a linkedin post !")
            return "linkedin"
        else:
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

        with st.spinner("Extracting the rescource..."):
            if self.classify_rescource(link) == "video":
                from src.aux_utils.transcription_utils import YouTubeTranscriber

                transcriber = YouTubeTranscriber()
                transcription = transcriber.transcribe(
                    input_path=link, method="groq"
                )
                self.current_rescource = (
                    "### This is a video transcription ### " + transcription
                )
                print("Rescource extracted successfully !")
                return transcription

            elif self.classify_rescource(link) == "linkedin":
                rescource = str(extract_linkedin(link))
                print("FULL RESCOURCE: ", rescource)
                self.current_rescource = (
                    "### This is a linkedin post ### " + rescource
                )
                print("Rescource extracted successfully !")
                return rescource
            else:
                return self.extract_website(link)
            
    def extract_rescource(self,text):
        """Extract the rescource from the given text. no link.
        Args:
            text (str): The text to be extracted.
        """
        self.current_rescource = text
        print("Rescource extracted successfully !")
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

        import time

        # First we save the rescource in a file in the appropriate folder
        from src.aux_utils.text_classification_utils import IntentClassifier

        with st.spinner("Indexing the resource..."):
            topic_classifier = IntentClassifier(config=self.config,
                labels_dict=self.config["data_sources"]
            )

            # get the associated topic description
            topic = topic_classifier.classify(
                self.current_rescource, method="LLM"
            )

            print("Topic Finded by classifier: ", topic)

            # get the directory associated to the topic (the key associated to the topic in the dictionary)
            directory = topic
            # save the rescource in the data/{directory} folder in .md format

            with open(
                f"data/{directory}/resource_{time.time()}.md",
                "w",
                encoding="utf-8",
            ) as file:
                file.write(self.current_rescource)
                print(
                    "Rescource saved successfully ! in the data/"
                    + directory
                    + " folder !"
                )

            # we index the rescource in the database !
            from src.main_utils.vectorstore_utils_v2 import VectorAgent
            # check if qdrant client is in session state

            vector_agent = VectorAgent(
                default_config=self.config, qdrant_client=self.client
            )
            vector_agent.fill()


if __name__ == "__main__":
    # load config using yaml from config/config.yaml
    # import yaml
    # with open('config/config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    # manager=ExternalKnowledgeManager(config)
    # link="https://youtu.be/_4ZoBEmcFXI?si=LLManhdxYnzJMc7K"
    # manager.extract_rescource_from_link(link)
    # manager.index_rescource()
    link = "https://www.linkedin.com/posts/yanis-labeyrie-67b11b225_openai-gpt5-o1-activity-7240116943750385664-uxzS?utm_source=share&utm_medium=member_desktop"
    content = extract_linkedin(link)
    print("Type of the content: ", type(content))
    print("##############################################")
    print(content)
    print("##############################################")
