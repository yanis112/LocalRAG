import logging
import os
import time
from functools import lru_cache

import streamlit as st
import yaml

# load environment variables
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError


# custom imports
from src.main_utils.LLM import CustomChatModel
from src.main_utils.retrieval_utils_v2 import RetrievalAgent
from src.main_utils.utils import (
    detect_language,
    log_execution_time,
)

load_dotenv()


# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename="rag_answer.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s",
)

def token_calculation_prompt(query: str) -> str:
    """
    Return the appropriate token calculation prompt for the query or document.
    """
    coefficient = 1 / 0.45
    num_tokens = len(query.split()) * coefficient
    return num_tokens

@lru_cache(maxsize=None)
def load_chat_model(model_name, temperature, llm_provider):
    """load the chat model from the model name and the temperature

    Args:
        model_name (str): the name of the model
        temperature (float): the temperature of the model
        llm_provider (str): the provider of the model

    Returns:
        CustomChatModel: the chat model object
    """
    chat_model_1 = CustomChatModel(
        llm_name=model_name, temperature=temperature, llm_provider=llm_provider
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
    
    logging.info("TEMPERATURE USED IN LLM CALL: %s", temperature)
    
    if llm_provider == 'github':
        token = os.environ["GITHUB_TOKEN"]
        endpoint = "https://models.inference.ai.azure.com"
        client = OpenAI(base_url=endpoint, api_key=token)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        if stream:
            response = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=temperature,
                stream=True
            )
            def stream_generator():
                for update in response:
                    if update.choices[0].delta.content:
                        yield update.choices[0].delta.content
            return stream_generator()
        
        else:
            response = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=temperature
            )
            return response.choices[0].message.content
    
    # Existing logic for other providers
    if llm_provider == 'ollama':
        pull_model(model_name)

    chat_model_1 = load_chat_model(model_name, temperature, llm_provider)
    chat = chat_model_1.chat_model
    prompt_template = chat_model_1.chat_prompt_template

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
            logging.error("JSON PARSING ERROR: %s", e)
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
        self.retrieval_agent = RetrievalAgent(default_config=default_config, config=config)
        self.client=self.retrieval_agent.client #client transmission to the RAGAgent

    @log_execution_time
    def RAG_answer(self, query,merged_config=None):
        if merged_config is None:
            merged_config = {**self.default_config, **self.config}
        logging.basicConfig(
            filename="rag_answer.log",
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
        )
        logging.info("FINAL CONFIG FILTERS: %s", merged_config["field_filter"])

        if merged_config['llm_provider'] == 'ollama':
            pull_model(merged_config["model_name"])

        detected_language = detect_language(query)
        logging.info("QUERY LANGUAGE: %s", detected_language)
        merged_config["prompt_language"] = detected_language

        language_flags = {
            "en": "🇬🇧",
            "fr": "🇫🇷",
        }
        flag_emoji = language_flags.get(detected_language, "🔍")
        st.toast(f"Language detected: {flag_emoji}", icon="🔍")

        useful_docs = self.retrieval_agent.query_database(query)
        print("Number of useful docs:", len(useful_docs))
        logging.info("NUMBER OF DOCS FROM QUERY DATABASE : %d", len(useful_docs))

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
                prompt = f"""Tu es un assistant IA conçu pour fournir des réponses détaillées, étape par étape. Tes réponses doivent suivre cette structure :
                Commence par une section <thinking>.
                À l'intérieur de la section thinking :
                a. Analyse brièvement la question et présente ton approche.
                b. Présente un plan clair d'étapes pour résoudre le problème.
                c. Utilise un processus de raisonnement "Chain of Thought" si nécessaire, en décomposant ton processus de réflexion en étapes numérotées.
                Inclue une section <reflection> pour chaque idée où tu :
                a. Révérifies ton raisonnement.
                b. Vérifies les erreurs ou omissions potentielles.
                c. Confirme ou ajuste ta conclusion si nécessaire.
                Assure-toi de fermer toutes les sections de réflexion.
                Termine la section thinking avec </thinking>.
                Fournis ta réponse finale dans une section <output>.
                Utilise toujours ces balises dans tes réponses. Sois minutieux dans tes explications, en montrant chaque étape de ton processus de réflexion. 
                Vise à être précis et logique dans ton approche, et n'hésite pas à décomposer les problèmes complexes en composants plus simples. 
                Ton ton doit être analytique et légèrement formel, en se concentrant sur la communication claire de ton processus de réflexion.
                N'oublie pas : les balises <thinking> et <reflection> doivent être fermées à la fin de chaque section.
                Assure-toi que toutes les balises <tags> sont sur des lignes séparées sans autre texte. Ne mets pas d'autre texte sur une ligne contenant une balise.
                Réponds à la question suivante : {query}. Pour y répondre, tu utiliseras les informations contenues dans les documents suivants (nom du doc et page) : \n\n {str_context}"""
            else:
                prompt = f"""You are an AI assistant designed to provide detailed, step-by-step responses. Your outputs should follow this structure:
                Begin with a <thinking> section.
                Inside the thinking section:
                a. Briefly analyze the question and outline your approach.
                b. Present a clear plan of steps to solve the problem.
                c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.
                Include a <reflection> section for each idea where you:
                a. Review your reasoning.
                b. Check for potential errors or oversights.
                c. Confirm or adjust your conclusion if necessary.
                Be sure to close all reflection sections.
                Close the thinking section with </thinking>.
                Provide your final answer in an <output> section.
                Always use these tags in your responses. Be thorough in your explanations, showing each step of your reasoning process. 
                Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. 
                Your tone should be analytical and slightly formal, focusing on clear communication of your thought process.
                Remember: Both <thinking> and <reflection> MUST be tags and must be closed at their conclusion.
                Make sure all <tags> are on separate lines with no other text. Do not include other text on a line containing a tag.
                Answer the following question: {query}. To answer it, you will use the information contained in the following documents (document name and page): \n\n {str_context}"""
        else:
            if merged_config['prompt_language'] == 'fr':
                prompt = f"""Réponds à la question suivante : {query} . Pour y répondre, tu utiliseras les informations contenues dans les documents suivants: 
                \n\n {str_context}. Ne cite pas les documents et répond en français, n'inclue pas de faits non mentionnés explicitement dans les documents dans ta réponse"""
            else:
                prompt = f"""Answer the following question: {query}. To answer it, you will use the information contained in the following documents: 
                \n\n {str_context}. Do not explicitly cite the documents and answer in English. Do not include facts not explicitly mentioned in the documents in your answer."""

        logging.info("TIME TO FORMAT CONTEXT: %f", time.time() - start_time)
        nb_tokens = token_calculation_prompt(prompt)
        logging.info("APPROXIMATE NUMBER OF TOKENS IN THE PROMPT: %d", nb_tokens)
        logging.info("PROMPT: %s", prompt)

        stream_generator = LLM_answer_v3(
            prompt,
            model_name=merged_config["model_name"],
            llm_provider=merged_config["llm_provider"],
            stream=merged_config["stream"],
            temperature=merged_config["temperature"],
        )

        #del self.retrieval_agent

        if merged_config["return_chunks"]:
            return stream_generator, list_content, list_sources
        else:
            return stream_generator

    @log_execution_time
    def advanced_RAG_answer(self, query):
        merged_config = {**self.default_config, **self.config}
        logging.basicConfig(
            filename="rag_answer.log",
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
        )
        logging.info("FINAL CONFIG FILTERS: %s", merged_config["field_filter"])

        if merged_config['llm_provider'] == 'ollama':
            pull_model(merged_config["model_name"])

        detected_language = detect_language(query)
        merged_config["prompt_language"] = detected_language

        chat_history = None
        if 'chat_history' in merged_config and merged_config['use_history']:
            chat_history = merged_config['chat_history']
            
        from src.main_utils.agentic_rag_utils import ChabotMemory, QueryBreaker, TaskTranslator

        memory = ChabotMemory(config=merged_config)
        task_translator = TaskTranslator(config=merged_config)
        query_breaker = QueryBreaker(config=merged_config)

        language_flags = {
            "en": "🇬🇧",
            "fr": "🇫🇷",
        }
        flag_emoji = language_flags.get(detected_language, "🔍")
        st.toast(f"Language detected: {flag_emoji}", icon="🔍")

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
                answer, docs, sources = self.RAG_answer(prompt)
                sources = [str(source) for source in sources]
                print("#################################################")
                print("INTERMEDIATE ANSWER:", answer)
                memory.ingest_info(f"{{STEP {step_number + 1}: {step} QUERY: {prompt} ANSWER: {answer}}}")
                step_number += 1
            list_sources.extend(sources)

        list_sources = list(set(list_sources))
        str_context = memory.get_content()
        if merged_config["cot_enabled"]:
            if merged_config['prompt_language'] == 'fr':
                prompt_final = f"""Tu es un assistant IA conçu pour fournir des réponses détaillées, étape par étape. Tes réponses doivent suivre cette structure :
                Commence par une section <thinking>.
                À l'intérieur de la section thinking :
                a. Analyse brièvement la question et présente ton approche.
                b. Présente un plan clair d'étapes pour résoudre le problème.
                c. Utilise un processus de raisonnement "Chain of Thought" si nécessaire, en décomposant ton processus de réflexion en étapes numérotées.
                Inclue une section <reflection> pour chaque idée où tu :
                a. Révérifies ton raisonnement.
                b. Vérifies les erreurs ou omissions potentielles.
                c. Confirme ou ajuste ta conclusion si nécessaire.
                Assure-toi de fermer toutes les sections de réflexion.
                Termine la section thinking avec </thinking>.
                Fournis ta réponse finale dans une section <output>.
                Utilise toujours ces balises dans tes réponses. Sois minutieux dans tes explications, en montrant chaque étape de ton processus de réflexion. 
                Vise à être précis et logique dans ton approche, et n'hésite pas à décomposer les problèmes complexes en composants plus simples. 
                Ton ton doit être analytique et légèrement formel, en se concentrant sur la communication claire de ton processus de réflexion.
                N'oublie pas : les balises <thinking> et <reflection> doivent être fermées à la fin de chaque section.
                Assure-toi que toutes les balises <tags> sont sur des lignes séparées sans autre texte. Ne mets pas d'autre texte sur une ligne contenant une balise.
                Basé sur les étapes intermédiaires et leurs réponses respectives, fournis une réponse finale à la question suivante : {query}, en utilisant les étapes de raisonnement intermédiaires et les réponses : {str_context}. Ne cite pas explicitement les étapes dans la réponse finale."""
            else:
                prompt_final = f"""You are an AI assistant designed to provide detailed, step-by-step responses. Your outputs should follow this structure:
                Begin with a <thinking> section.
                Inside the thinking section:
                a. Briefly analyze the question and outline your approach.
                b. Present a clear plan of steps to solve the problem.
                c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.
                Include a <reflection> section for each idea where you:
                a. Review your reasoning.
                b. Check for potential errors or oversights.
                c. Confirm or adjust your conclusion if necessary.
                Be sure to close all reflection sections.
                Close the thinking section with </thinking>.
                Provide your final answer in an <output> section.
                Always use these tags in your responses. Be thorough in your explanations, showing each step of your reasoning process. 
                Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. 
                Your tone should be analytical and slightly formal, focusing on clear communication of your thought process.
                Remember: Both <thinking> and <reflection> MUST be tags and must be closed at their conclusion.
                Make sure all <tags> are on separate lines with no other text. Do not include other text on a line containing a tag.
                Based on the intermediate steps and their respective answers, provide a final answer to the following query: {query}, using the intermediate reasoning steps and answers: {str_context}. Do not explicitly cite the steps in the final answer."""
        else:
            if merged_config['prompt_language'] == 'fr':
                prompt_final = f"""Réponds à la question suivante : {query}. Pour y répondre, tu utiliseras les informations contenues dans les étapes intermédiaires et leurs réponses respectives : {str_context}. Ne cite pas explicitement les étapes dans la réponse finale et répond en français."""
            else:
                prompt_final = f"""Answer the following question: {query}. To answer it, you will use the information contained in the intermediate steps and their respective answers: {str_context}. Do not explicitly cite the steps in the final answer and answer in English."""

        with st.spinner("Working on the final answer... 🤔⚙️ "):
            stream_generator = LLM_answer_v3(
                prompt_final,
                model_name=merged_config["model_name"],
                llm_provider=merged_config["llm_provider"],
                stream=merged_config["stream"],
                temperature=merged_config["temperature"],
            )

        return stream_generator, list_sources
    
    
    
    def internet_rag(self,query):
        """Return an answer to the user's query using the RAG model with the internet as the source of information.

        Args:
            query (str): The user's query.
        Returns:
            str: The answer to the user's query.
            
        """
        #load internet_config
        with open("config/internet_config.yaml") as f:
            internet_config = yaml.safe_load(f)
            
        #create a merged config using self.merge_config and internet_config
        merged_config = {**self.default_config,**self.config}
            
        #make a new merged config using self.merge_config and internet_config
        internet_merged_config = {**merged_config,**internet_config}
        
        from src.internet_utils import InternetAgent
        internet_agent = InternetAgent()
        
        #we empty the internet folder of its contents
        import shutil
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
        #agent.fill()
        list_pages=agent.get_chunks()
        
        print("Number of chunks created:",len(list_pages))
            
        # Embed the documents using SentenceTransformer
        model = self.load_embedding_model()
        # Python
        
        list_pages_text = [page.page_content for page in list_pages]
        doc_embeddings = model.encode(list_pages_text, convert_to_tensor=True,device='cuda')
        #doc_embeddings = model.encode(list_pages, convert_to_tensor=True)
        query_embedding = model.encode([query], convert_to_tensor=True,device='cuda')
        
        # Compute the similarity between the query and the documents
        similarities = model.similarity(query_embedding, doc_embeddings)
        topk_values, topk_indices = similarities.topk(5)
        
        # Retrieve the top-k documents based on similarity
        topk_docs = [list_pages[idx] for idx in topk_indices[0]]
        
        # Create the prompt using the context documents
        context = "\n\n".join([doc.page_content for doc in topk_docs])
        prompt = f"""
        You are an AI assistant. Use the following documents as context to answer the user's question.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:
        """
        
        #query the vectorstore with RAG_answer method
        if self.config["return_chunks"]:
            #add return_chunks to the config
      
             # Call the LLM_answer_v3 function with the prompt
            answer = LLM_answer_v3(prompt,model_name=internet_merged_config["model_name"],stream=internet_merged_config["stream"],llm_provider=internet_merged_config["llm_provider"])
        
            #stream_generator, list_content, list_sources = self.RAG_answer(query,merged_config=internet_config)
            #delete the vectorstore (its persist directory)
            #agent.delete()
            #delete the internet folder
            # import shutil
            # shutil.rmtree("data/internet")
            return answer,topk_docs
            
        else:
            #add return_chunks to the config

            # Call the LLM_answer_v3 function with the prompt
            answer = LLM_answer_v3(prompt,model_name=internet_merged_config["model_name"],stream=internet_merged_config["stream"],llm_provider=internet_merged_config["llm_provider"])
            #stream_generator = self.RAG_answer(query,merged_config=internet_config)
            #agent.delete()
            #delete the internet folder
            # import shutil
            # shutil.rmtree("data/internet")
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

    query = "Qui est Simon Boiko ?"
    agent = RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})
    answer = agent.RAG_answer(query)
    print("Answer:", answer)
    exit()
    #test internet rag
    agent = RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})
    query = "Est il vrai que Elon Musk a proposé de l'argent à des américains pour qu'ils votent pour Trump explicitement ?"
    start_time = time.time()
    stream_generator = agent.internet_rag(query)
    end_time = time.time()
    
    #reconstruct the answer from stream_generator
    print("Answer:")
 
    print('stream_generator:', stream_generator)
    # for chunk in stream_generator:
    #     print(chunk)
    
    
    print("Time taken:", end_time - start_time)

