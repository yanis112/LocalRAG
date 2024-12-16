
import os
import time
from functools import lru_cache

import streamlit as st
import yaml

# load environment variables
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from langchain_core.prompts import PromptTemplate


# custom imports
from src.aux_utils.logging_utils import setup_logger
from src.main_utils.LLM import CustomChatModel
from src.main_utils.retrieval_utils_v2 import RetrievalAgent
from src.main_utils.utils import (
    detect_language,
    log_execution_time,
)

load_dotenv()


# Create logger directory if it doesn't exist
# log_dir = "logger"
# os.makedirs(log_dir, exist_ok=True)

# # Configure logger
# for handler in logger.root.handlers[:]:
#     logger.root.removeHandler(handler)

# logger.basicConfig(
#     filename=os.path.join(log_dir, "rag_answer.log"),
#     filemode="a",
#     level=logger.INFO,
#     format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s",
# )

logger = setup_logger(
    __name__, 
    "rag_answer.log"
)


def token_calculation_prompt(query: str) -> str:
    """
    Return the appropriate token calculation prompt for the query or document.
    """
    coefficient = 1 / 0.45
    num_tokens = len(query.split()) * coefficient
    return num_tokens

@lru_cache(maxsize=None)
def load_chat_model(model_name, temperature, llm_provider,system_prompt=None):
    """load the chat model from the model name and the temperature

    Args:
        model_name (str): the name of the model
        temperature (float): the temperature of the model
        llm_provider (str): the provider of the model

    Returns:
        CustomChatModel: the chat model object
    """
    chat_model_1 = CustomChatModel(
        llm_name=model_name, temperature=temperature, llm_provider=llm_provider,system_prompt=system_prompt
    )
    return chat_model_1


@lru_cache(maxsize=None)
def pull_model(model_name):
    """
    Pulls a model using the ollama CLI.

    Args:
        model_name (str): The name of the model to pull.
    return:
        str: 'model pulled !' if the model was successfully pulled, 'model not pulled !' otherwise.
    """
    import subprocess
    try:
        with st.spinner("Model not loaded yet, pulling the model... ⚙️"):
            # Construct the command
            command = ['ollama', 'pull', model_name]
            
            # Execute the command
            
            result = subprocess.run(command, check=True, capture_output=False, text=True)
            
            # Print the output
            print(result.stdout)
    except Exception as e:
        print(f"An error occurred while pulling the model: {e}, maybe check the model name is correctly spelled !")
    
    return 'model pulled !'


@log_execution_time
def LLM_answer_v3(
    prompt,
    json_formatting=False,
    pydantic_object=None,
    format_type=None,
    model_name=None,
    temperature=1,
    stream=False,
    llm_provider=None,
    system_prompt=None,
):
    """
    Generates an answer using a specified language model (LLM) based on the provided prompt and configuration parameters.
    Args:
        prompt (str): The input prompt to generate an answer for.
        json_formatting (bool, optional): If True, the output will be formatted as JSON. Defaults to False.
        pydantic_object (BaseModel, optional): A Pydantic model class used for JSON formatting. Required if json_formatting is True.
        format_type (str, optional): Specifies the format type (e.g., 'list' or 'dict') for structured output. Defaults to None.
        model_name (str, optional): The name of the model to use. Defaults to None.
        temperature (float, optional): The temperature setting for the model, affecting the randomness of the output. Defaults to 1.
        stream (bool, optional): If True, the output will be streamed. Defaults to False.
        llm_provider (str, optional): The provider of the language model (e.g., 'ollama'). Defaults to None.
    Returns:
        str or None: The generated answer from the language model. If json_formatting is True and an error occurs during JSON parsing, returns None.
    Raises:
        ValidationError: If an error occurs during JSON parsing when json_formatting is True.
    """
    
    logger.info("TEMPERATURE USED IN LLM CALL: %s", temperature)
    
    # Existing logic for other providers
    if llm_provider == 'ollama':
        pull_model(model_name)

    llm = load_chat_model(model_name, temperature, llm_provider)
    if system_prompt is not None:
        llm.system_prompt = system_prompt
    #define the chat and prompt_template
    chat = llm.chat_model   
    prompt_template = llm.chat_prompt_template
    

    if json_formatting and issubclass(pydantic_object, BaseModel):
        from langchain_core.output_parsers import JsonOutputParser
        parser = JsonOutputParser(pydantic_object=pydantic_object)
        format_instructions = parser.get_format_instructions()
        if format_type:
            from src.main_utils.utils import get_strutured_format
            format_instructions = get_strutured_format(format_type)
            schema = parser._get_schema(pydantic_object)
            format_instructions = format_instructions + "```" + str(schema) + "```"
        
        total_prompt = "Answer the user query. \n" + str(prompt) + "\n" + str(format_instructions)
        chain = prompt_template | chat | parser
        try:
            result = chain.invoke({"text": total_prompt})
            return result
        except ValidationError as e:
            logger.error("JSON PARSING ERROR: %s", e)
            return None
    else:
        from langchain.schema import StrOutputParser
        chain = prompt_template | chat | StrOutputParser()
        if stream:
            return chain.stream({"text": prompt})
        else:
            return chain.invoke({"text": prompt})

class RAGAgent:
    def __init__(self, default_config, config={"stream": False}):
        self.default_config = default_config
        self.config = config
        self.merged_config = {**default_config, **config}
        self.retrieval_agent = RetrievalAgent(default_config=default_config, config=config)
        self.client=self.retrieval_agent.client #client transmission to the RAGAgent
        self.system_prompt=None

    @log_execution_time
    def RAG_answer(self, query,merged_config=None,system_prompt=None):
        """
        Generate an answer to a given query using a Retrieval-Augmented Generation (RAG) approach.
        This method processes the query, retrieves relevant documents from a database, and generates a response
        using a language model. The response can be formatted with specific prompts and configurations.
        Args:
            query (str): The input query for which an answer is to be generated.
            merged_config (dict, optional): Configuration dictionary for the RAG process. If None, the default
                configuration is used. Defaults to None.
            system_prompt (str, optional): An optional system prompt to guide the language model's response.
                Defaults to None.
        Returns:
            generator: A generator that yields the language model's response in chunks.
            list: A list of document contents used for generating the response (if `return_chunks` is True).
            list: A list of document sources used for generating the response (if `return_chunks` is True).
        Raises:
            ValueError: If the language model provider specified in the configuration is not supported.
        Notes:
            - The method detects the language of the query and adjusts the prompt language accordingly.
            - It supports both English and French languages for the Chain of Thought (CoT) reasoning process.
            - The method logs various stages of the process, including the number of retrieved documents and
                the time taken to format the context.
        """
        if merged_config is None:
            merged_config = self.merged_config
        
        logger.info("FINAL CONFIG FILTERS: %s", merged_config["field_filter"])

        if merged_config['llm_provider'] == 'ollama':
            pull_model(merged_config["model_name"])

        detected_language = detect_language(query)
        logger.info("QUERY LANGUAGE: %s", detected_language)
        merged_config["prompt_language"] = detected_language

        language_flags = {
            "en": "🇬🇧",
            "fr": "🇫🇷",
        }
        flag_emoji = language_flags.get(detected_language)
        st.toast(f"Language detected: {flag_emoji}", icon=flag_emoji)

        useful_docs = self.retrieval_agent.query_database(query)
        print("Number of useful docs:", len(useful_docs))
        logger.info("NUMBER OF DOCS FROM QUERY DATABASE : %d", len(useful_docs))

        start_time = time.time()
        list_content = [doc.page_content for doc in useful_docs]
        list_metadata = [doc.metadata for doc in useful_docs]
        list_sources = [metadata["source"] for metadata in list_metadata]

        list_context = [
            f"Document {i}: {content} \n\n "
            for i, (metadata, content) in enumerate(zip(list_metadata, list_content))
        ]

        if 'use_history' in merged_config and merged_config['use_history']:
            list_context.append(f"Document {len(list_metadata)}: CHAT HISTORY (query asked and answered before the current one): {merged_config['chat_history']}")

        str_context = " ".join(list_context)
        if merged_config["cot_enabled"]:
            if merged_config['prompt_language'] == 'fr':
                #load the cot prompt from the prompts folder
                with open("prompts/cot_prompt_fr.txt") as f:
                    prompt=f.read()
                
                #format using prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                prompt = prompt_template.format(query=query,context=str_context)
                
            else:
                with open("prompts/cot_prompt_en.txt") as f:
                    prompt=f.read()
                    
                #format using prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                prompt = prompt_template.format(query=query,context=str_context)
        else:
            if merged_config['prompt_language'] == 'fr':
                with open("prompts/base_rag_prompt_fr.txt") as f:
                    prompt=f.read()
                #format using prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                prompt = prompt_template.format(query=query,context=str_context)
            else:
                with open("prompts/base_rag_prompt_en.txt") as f:
                    prompt=f.read()
                #format using prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                prompt = prompt_template.format(query=query,context=str_context)
                    
        logger.info("TIME TO FORMAT CONTEXT: %f", time.time() - start_time)
        nb_tokens = token_calculation_prompt(prompt)
        logger.info("APPROXIMATE NUMBER OF TOKENS IN THE PROMPT: %d", nb_tokens)
        logger.info("PROMPT: %s", prompt)

        stream_generator = LLM_answer_v3(
            prompt,
            model_name=merged_config["model_name"],
            llm_provider=merged_config["llm_provider"],
            stream=merged_config["stream"],
            temperature=merged_config["temperature"],
            system_prompt=system_prompt
        )

        if merged_config["return_chunks"]:
            return stream_generator, list_content, list_sources
        else:
            return stream_generator

    @log_execution_time
    def advanced_RAG_answer(self, query, system_prompt=None):
        
       #we turn off the stream for the advanced RAG answer
        self.merged_config["stream"] = False
        logger.info("FINAL CONFIG FILTERS: %s", self.merged_config["field_filter"])

        if self.merged_config['llm_provider'] == 'ollama':
            pull_model(self.merged_config["model_name"])

        detected_language = detect_language(query)
        self.merged_config["prompt_language"] = detected_language
        
        
        #we want to return the sources
        self.merged_config["return_chunks"] = True

        chat_history = None
        if 'chat_history' in self.merged_config and self.merged_config['use_history']:
            chat_history = self.merged_config['chat_history']
            
        from src.main_utils.agentic_rag_utils import ChabotMemory, QueryBreaker, TaskTranslator

        memory = ChabotMemory(config=self.merged_config) #memory to store the context of the reflexion
        task_translator = TaskTranslator(config=self.merged_config) #translator to translate reflexion steps into prompts for the rag
        query_breaker = QueryBreaker(config=self.merged_config) #query breaker to break the query into reflexion steps

        language_flags = {
            "en": "🇬🇧",
            "fr": "🇫🇷",
        }
        flag_emoji = language_flags.get(detected_language)
        st.toast("Language detected", icon=flag_emoji)

        with st.spinner("Getting intermediate thinking steps..."):
            intermediate_steps = query_breaker.break_query(query, context=chat_history)
            intermediate_steps = [str(step) for step in intermediate_steps]

        st.toast("Intermediate steps computed!")
        print("#################################################")
        print("INTERMEDIATE STEPS LIST:", intermediate_steps)

        list_sources = []
        step_number = 0
        for step in intermediate_steps:
            print("#################################################")
            print("TREATING STEP NUMBER:", step_number)
            with st.spinner(f"🔄 Working on step {step_number + 1} out of {len(intermediate_steps)}..."):
                prompt = task_translator.get_prompt(step, context=memory.get_content())
                print("#################################################")
                print("TASK PROMPT:", prompt)
            with st.spinner(f"🤖 Answering intermediate query: {prompt}"):
                answer, docs, sources = self.RAG_answer(prompt, system_prompt=system_prompt)
                sources = [str(source) for source in sources]
                print("#################################################")
                print("INTERMEDIATE ANSWER:", answer)
                memory.ingest_info(f"{{STEP {step_number + 1}: {step} QUERY: {prompt} ANSWER: {answer}}}")
                step_number += 1
            list_sources.extend(sources)

        list_sources = list(set(list_sources))
        str_context = memory.get_content()
        if self.merged_config["cot_enabled"]:
            if self.merged_config['prompt_language'] == 'fr':
                with open("prompts/cot_prompt_fr.txt") as f:
                    prompt=f.read()
                #format using prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                prompt = prompt_template.format(query=query,context=str_context)
            else:
                with open("prompts/cot_prompt_en.txt") as f:
                    prompt=f.read()
                #format using prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                prompt = prompt_template.format(query=query,context=str_context)
        else:
            if self.merged_config['prompt_language'] == 'fr':
                with open("prompts/base_rag_prompt_fr.txt") as f:
                    prompt=f.read()
                #format using prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                prompt = prompt_template.format(query=query,context=str_context)
            else:
                with open("prompts/base_rag_prompt_en.txt") as f:
                    prompt=f.read()
                #format using prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                prompt = prompt_template.format(query=query,context=str_context)
                
        with st.spinner("Working on the final answer... 🤔⚙️ "):
            #we turn the streaming back on for the final answer
            #turn back the stream to True
            self.merged_config["stream"] = True
            stream_generator = LLM_answer_v3(
                prompt,
                model_name=self.merged_config["model_name"],
                llm_provider=self.merged_config["llm_provider"],
                stream=self.merged_config["stream"],
                temperature=self.merged_config["temperature"],
                system_prompt=self.system_prompt
            )

        return stream_generator, list_sources
    
    
    def internet_rag(self,query):
        """Return an answer to the user's query using the RAG model with the internet as the source of information.

        Args:
            query (str): The user's query.
        Returns:
            str: The answer to the user's query.
            
        """
        from src.aux_utils.internet_utils import InternetAgent
        import shutil
        #load internet_config
        with open("config/internet_config.yaml") as f:
            internet_config = yaml.safe_load(f)

        #make a new merged config using self.merge_config and internet_config
        internet_merged_config = {**self.merged_config,**internet_config}
        
        internet_agent = InternetAgent()
        
        #we empty the internet folder of its contents
        shutil.rmtree("data/internet")
        
        #create an internet folder in the data folder if it does not exist already
        if not os.path.exists("data/internet"):
            os.makedirs("data/internet")
        
        #get urls
        urls = internet_agent.get_urls(query, num_results=internet_merged_config["num_urls"])
        print("Number of urls:",len(urls))
        
        #scrape contents
        list_pages = internet_agent.scrape_contents(urls)
        print("Number of pages scraped:",len(list_pages))
        #save the content of the pages in the internet folder
        internet_agent.save_pages()

        print("Contents downloaded in html format")
        
        #create temporary vectorstore to store the pages
        from src.main_utils.vectorstore_utils_v2 import VectorAgent
        # Create a VectorAgent object
        agent = VectorAgent(default_config=internet_merged_config)
        # Fill the vectorstore with the pages
        agent.fill()
        
        #answer to the query using RAG_answer
        answer = self.RAG_answer(query,merged_config=internet_merged_config)
        
        return answer
        
    @lru_cache(maxsize=None)
    def load_embedding_model(self):
        """_summary_
        """
        from sentence_transformers import SentenceTransformer
        # Load the SentenceTransformer model
        model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True,device='cuda')
        return model
        

if __name__ == "__main__":
    import time
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # query = "Qui est Simon Boiko ?"
    # agent = RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})
    # answer = agent.RAG_answer(query)
    # print("Answer:", answer)
    # exit()
    #test internet rag
    agent = RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})
    query = "Est il vrai que Elon Musk a proposé de l'argent à des américains pour qu'ils votent pour Trump explicitement ?"
    start_time = time.time()
    stream_generator = agent.internet_rag(query)
    end_time = time.time()
    print("FINAL_ANSWER:",stream_generator)
    
    # #reconstruct the answer from stream_generator
    # print("Answer:")
 
    # print('stream_generator:', stream_generator)
    # # for chunk in stream_generator:
    # #     print(chunk)
    
    
    # print("Time taken:", end_time - start_time)

