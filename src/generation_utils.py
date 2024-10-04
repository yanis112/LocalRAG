import logging
import os
import time
from functools import lru_cache

import yaml

# load environment variables
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel
from langsmith import traceable
import streamlit as st

# custom imports
from src.LLM import CustomChatModel
from src.retrieval_utils import query_database_v2, query_knowledge_graph_v2, query_knowledge_graph_v3
from src.utils import (
    log_execution_time,
    token_calculation_prompt,
    translate_to_french,
    translate_to_english,
    detect_language
)


from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from pydantic import BaseModel, ValidationError
import subprocess



load_dotenv()


class Trust(BaseModel):
    score: str


os.environ["LANGCHAIN_TRACING_V2"] = "true"  # UTILE !!
os.environ["LANGCHAIN_API_KEY"] = (
    "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"
)
os.environ["LANGCHAIN_END_POINT"] = "https://api.smith.langchain.com"

# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename="rag_answer.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s",
)



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
def LLM_answer(
    prompt,
    json_formatting=False,
    pydantic_object=None,
    model_name=None,
    temperature=1,
    stream=False,
    llm_provider=None,
):  #DEPRECATED
    """
    Generate an answer using the Language Model (LLM) based on the given prompt.

    Args:
        prompt (str): The input prompt for generating the answer.
        json_formatting (bool, optional): Flag indicating whether the output should be in JSON format. Defaults to False.
        pydantic_object (Any, optional): The Pydantic object used for JSON formatting. Defaults to None.
        model_name (str, optional): The name of the language model to use. Defaults to None.
        temperature (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to 1.
        stream (bool, optional): Flag indicating whether to stream the output. Defaults to False.
        llm_provider (Any, optional): The provider for the language model. Defaults to None.

    Returns:
        Any: The generated answer.

    Raises:
        Exception: If an error occurs during JSON parsing.

    """
    # Initialize the result
    result = None

    logging.info("TEMPERATURE USED IN LLM CALL: %s", temperature)
    
    #we try to pull the model if it is not already pulled
    
    #pull_model(model_name)

    # we create the chat model object with contains a LLM and a prompt template and a chat model (LLM+prompt template)
    chat_model_1 = load_chat_model(model_name, temperature, llm_provider)

    # we get the chat model / LLM from the chat model object
    chat = chat_model_1.chat_model

    prompt_template = chat_model_1.chat_prompt_template

    if json_formatting:
        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=pydantic_object)
        prompt_template = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{prompt}\n",  # template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )
        
        #print the format instructions 
        #print("FORMAT INSTRUCTIONS:", parser.get_format_instructions())
        
        
        chain = prompt_template | chat | parser
        try:
            result = chain.invoke({"query": prompt})
            #print("RAW RESULT:", result)
            return result
        except Exception as e:
            print("ERROR OCCURED DURING JSON PARSING !")
            logging.error("ERROR OCCURED DURING JSON PARSING !: %s", e)

    else:
        chain = prompt_template | chat | StrOutputParser()
        if stream:
            stream_generator = chain.stream(
                {"text": prompt}
            )  
            return stream_generator
        else:
            result = chain.invoke(
                {"text": prompt}
            ) 
            return result
        
  
format_instruction="""
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"query1": {"title": "Query1", "type": "string"}, "query2": {"title": "Query2", "type": "string"}, "query3": {"title": "Query3", "type": "string"}, "query4": {"title": "Query4", "type": "string"}, "query5": {"title": "Query5", "type": "string"}}, "required": ["query1", "query2", "query3", "query4", "query5"]}
```
"""

@traceable
@log_execution_time
def LLM_answer_v2( #DEPRECATED
    prompt,
    json_formatting=False,
    pydantic_object=None,
    model_name=None,
    temperature=1,
    stream=False,
    llm_provider=None,
):
    logging.info("TEMPERATURE USED IN LLM CALL: %s", temperature)

    chat_model_1 = load_chat_model(model_name, temperature, llm_provider)
    chat = chat_model_1.chat_model
    prompt_template = chat_model_1.chat_prompt_template

    if json_formatting and issubclass(pydantic_object, BaseModel):
        parser = JsonOutputParser(pydantic_object=pydantic_object)
        format_instructions = parser.get_format_instructions()
        #get the shema: _get_schema
        #print("SCHEMA:", parser._get_schema(pydantic_object))
        #print("FORMAT INSTRUCTIONS:", format_instructions)
        total_prompt="Answer the user query.\n"+str(format_instructions) + str(prompt) + "\n"   #we create the total prompt by adding the format instructions to the prompt
        #print('TOTAL PROMPT:', total_prompt)
        chain = prompt_template | chat | parser #we create the chain with parser
        try:
            result = chain.invoke({"text": total_prompt})
            #print("RAW RESULT:", result)
            return result
        except ValidationError as e:
            print("ERROR OCCURRED DURING JSON PARSING!")
            logging.error("JSON PARSING ERROR: %s", e)
            return None
    else:
        chain = prompt_template | chat | StrOutputParser()
        if stream:
            #print("Stream detected")
            return chain.stream({"text": prompt})
        else:
            #print("No stream detected")
            return chain.invoke({"text": prompt})

@traceable
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
    
    #we try to pull the model if it is not already pulled
    if llm_provider=='ollama':
        pull_model(model_name)

    #we load appropriate templates and models
    chat_model_1 = load_chat_model(model_name, temperature, llm_provider)
    chat = chat_model_1.chat_model
    prompt_template = chat_model_1.chat_prompt_template
    
    

    if json_formatting and issubclass(pydantic_object, BaseModel):
        parser = JsonOutputParser(pydantic_object=pydantic_object)
        format_instructions = parser.get_format_instructions()
        if format_type:
            from src.utils import get_strutured_format

            format_instructions=get_strutured_format(format_type) #list or dict
            schema=parser._get_schema(pydantic_object)
            
            # we add the schema betwee, ``` and ``` to the format_instructions
            format_instructions=format_instructions + "```"+str(schema)+"```"
            
            
        #get the shema: _get_schema
        #print("SCHEMA:", parser._get_schema(pydantic_object))
        #print("FORMAT INSTRUCTIONS:", format_instructions)
        
        total_prompt="Answer the user query. \n"+str(prompt) + "\n"  + str(format_instructions)    #we create the total prompt by adding the format instructions to the prompt
        print("##############################################################")
        #print('TOTAL PROMPT:', total_prompt)
        chain = prompt_template | chat | parser #we create the chain with parser
        try:
            result = chain.invoke({"text": total_prompt})
            #print("RAW RESULT:", result)
            return result
        except ValidationError as e:
            print("ERROR OCCURRED DURING JSON PARSING!")
            logging.error("JSON PARSING ERROR: %s", e)
            return None
    else:
        chain = prompt_template | chat | StrOutputParser()
        if stream:
            #print("Stream detected")
            return chain.stream({"text": prompt})
        else:
            #print("No stream detected")
            return chain.invoke({"text": prompt})

def context_splitter(context, max_tokens=1000):
    """
    Function that takes the context as input and splits it into chunks of max_tokens tokens.
    Input:
        context: the context to split. type: string
        max_tokens: the maximum number of tokens in each chunk. type: int
    Output:
        list_chunks: The list of chunks
    """
    # Tokenize the context (simple split by spaces, can be improved with NLP libraries for better tokenization)
    print("FULL CONTEXT:", context)
    tokens = context.split()
    list_chunks = []
    i = 0
    # make correspondance between max tokens and max words
    max_tokens = round(max_tokens * 1.7)
    # Split the tokens into chunks of max_tokens tokens
    for i in range(0, len(tokens), max_tokens):
        list_chunks.append(" ".join(tokens[i : i + max_tokens]))

    return list_chunks



def node_dict_to_paragraph(input_dict):
    """
    Converts a dictionary of representing a retrieved node from the knowledge graph to a paragraph string.

    Args:
        input_dict (dict): A dictionary containing nodes.

    Returns:
        str: A paragraph string generated from the input dictionary.
    """
    paragraph = ""
    for document in input_dict:
        page_content = (
            document.page_content
        )  # Adjusted to access 'page_content' correctly
        if (
            "relation_list" in document.metadata
        ):  # Adjusted to access 'metadata' correctly
            paragraph += "Relations concerning the entity: \n"
            for relation in document.metadata["relation_list"]:
                # {"r1":"works_for","r2":"contributes_to","r3":"works_with","r4":"makes_use_of","r5":"located_in","r6":"included_in"}
                relation_phrase = (
                    relation["relation"]
                    .replace("workswith", "works with")
                    .replace("workson", "works on")
                    .replace("contributesto", "contributes to")
                    .replace("makesuseof", "makes use of")
                    .replace("locatedin", "located in")
                    .replace("includedin", "included in")
                    .replace("workfor", "works for")
                )
                paragraph += f"{page_content} {relation_phrase} {relation['linked_entity']}. "

        if "description" in document.metadata:
            paragraph += (
                f"Entity description: {document.metadata['description']}. "
            )
        if "community" in document.metadata:
            print("Community:", document.metadata["community"])
            paragraph += f"Entity belongs to the community: {document.metadata['community']['summary']}."

    return paragraph.strip()


# Configure logging


@log_execution_time
@traceable
def RAG_answer(query, default_config, config={"stream": False}):
    """
    Answer a given query using the RAG (Retrieval-Augmented Generation) model.
    Args:
        query (str): The query to be answered.
        default_config (dict): The default configuration settings.
        config (dict, optional): Additional configuration settings. Defaults to {"stream": False}.
        
    Returns:
        generator or tuple: (tuple) If `return_chunks` is True in the configuration, returns a generator for streaming the answer, along with the list of document contents and sources used. Otherwise, returns a generator for streaming the answer.
    """

    # Merge default_config and config. If a key exists in both, value from config is used.
    merged_config = {**default_config, **config}

    # Configure logging
    logging.basicConfig(
        filename="rag_answer.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    # Log merged config field filters
    logging.info("FINAL CONFIG FILTERS: %s", merged_config["field_filter"])
    
    #We try to pull the model if it is not already pulled
    if merged_config['llm_provider']=='ollama':
        pull_model(merged_config["model_name"])
    
    # Detect the language of the query
    detected_language = detect_language(query)

    logging.info("QUERY LANGUAGE: %s", detected_language)

    merged_config["prompt_language"] = detected_language

    # Mapping of languages to their respective flag emojis
    language_flags = {
        "en": "🇬🇧",  # English
        "fr": "🇫🇷",  # French
        # Add more languages and their respective flag emojis as needed
    }

    # Get the flag emoji for the detected language
    flag_emoji = language_flags[detected_language]  # Default to magnifying glass if language not found

    # Make a Streamlit toast
    st.toast(f"Language detected: {flag_emoji}", icon="🔍")
        
    # we search the useful docs in the database
    useful_docs = query_database_v2(query, default_config, config)

    if merged_config[
        "allow_kg_retrieval"
    ]:  # allow the retrieval of the knowledge graph information
        usefull_info = query_knowledge_graph_v3(
            query, default_config, config
        )  # this is a dictionary of str lists

    logging.info("NUMBER OF DOCS FROM QUERY DATABASE : %d", len(useful_docs))

    start_time = time.time()
    # We make a str using the document content and the metadata
    list_content = [doc.page_content for doc in useful_docs]

    # we list the metadata of the documents
    list_metadata = [doc.metadata for doc in useful_docs]
    # We list the sources of the documents
    list_sources = [metadata["source"] for metadata in list_metadata]

    # We create a context using all the documents and \n\n as a separator
    list_context = [
        f"Document {i}: {content} \n\n "
        for i, (metadata, content) in enumerate(
            zip(list_metadata, list_content)
        )
    ]
    
    #If the paramter use_history is set to True in the config, we add the chat history to the context
    if 'use_history' in merged_config and merged_config['use_history']:
        list_context.append(f"Document {len(list_metadata)}: CHAT HISTORY (query asked and answered before the current one): {merged_config['chat_history']}")
           

    str_context = " ".join(list_context)

    # if we allow the retrieval of the knowledge graph information we add it to the context
    if merged_config["allow_kg_retrieval"]:
        
        
        str_context += f"Document {len(list_metadata)}:"
        # add usefull relations
        str_context += (
            " Important relations concerning the entities mentionned: \n  "
        )
        
        if usefull_info["selected_relations"]:
            for relation in usefull_info["selected_relations"]:
                str_context += relation + " ;"
            # add the descriptions
            str_context += (
                " Descriptions of the entities mentionned in the query : \n"
            )
        #only if the description is not empty
        if usefull_info["selected_descriptions"]:
            for description in usefull_info["selected_descriptions"]:
                str_context += description + " ;"
                
        # add the communities
        #only if the community is not empty
        if usefull_info["selected_communities"]:
            str_context += "\n\n Entities mantionned in the query belongs to the following communities: \n"
            for community in usefull_info["selected_communities"]:
                str_context += community + " ;"
                
        #ajoute un saut de ligne à la fin
        str_context += "\n\n"

    if merged_config["cot_enabled"]:
        # We define a COT prompt for the LLM model
        if merged_config['prompt_language'] == 'fr':
            # CoT prompt with the provided structure entirely in French
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
            # CoT prompt with the provided structure in English
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
        # We define a prompt for the LLM model without CoT reasoning
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

    # We get the answer using the LLM model
    stream_generator = LLM_answer_v3(
        prompt,
        model_name=merged_config["model_name"],
        llm_provider=merged_config["llm_provider"],
        stream=merged_config["stream"],
        temperature=merged_config["temperature"],
    )
    if merged_config["return_chunks"]:
        return stream_generator, list_content, list_sources
    else:
        return stream_generator


@log_execution_time
@traceable
def advanced_RAG_answer(query, default_config, config={"stream": False}):
    """
    Answer a given query using the RAG (Retrieval-Augmented Generation) model by decomposing the query into intermediate steps and using the intermediate steps to generate the final answer.
    Args:
        query (str): The query to be answered.
        default_config (dict): The default configuration settings.
        config (dict, optional): Additional configuration settings. Defaults to {"stream": False}.
    Returns:
        generator or tuple: If `return_chunks` is True in the configuration, returns a generator for streaming the answer, along with the list of document contents and sources used. Otherwise, returns a generator for streaming the answer.
    Raises:
        None
    """

    # Merge default_config and config. If a key exists in both, value from config is used.
    merged_config = {**default_config, **config}

    # Configure logging
    logging.basicConfig(
        filename="rag_answer.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    # Log merged config field filters
    logging.info("FINAL CONFIG FILTERS: %s", merged_config["field_filter"])
    
    # We try to pull the model if it is not already pulled
    if merged_config['llm_provider'] == 'ollama':
        pull_model(merged_config["model_name"])
    
    # Load the memory object
    from src.agentic_rag_utils import ChabotMemory
    
    # Detect the language of the query
    detected_language = detect_language(query)
    
    # Modify the prompt language in the config before giving it to the memory object
    merged_config["prompt_language"] = detected_language
    
    # Check if a context is present in the config (the chatbot history)
    chat_history = None
    if 'chat_history' in merged_config and merged_config['use_history']:
        chat_history = merged_config['chat_history']
        
    memory = ChabotMemory(config=merged_config)
    
    # Load the TaskTranslator object
    from src.agentic_rag_utils import TaskTranslator
    task_translator = TaskTranslator(config=merged_config)
    
    # Load the QueryBreaker object
    from src.agentic_rag_utils import QueryBreaker
    query_breaker = QueryBreaker(config=merged_config)
  
    # Mapping of languages to their respective flag emojis
    language_flags = {
        "en": "🇬🇧",  # English
        "fr": "🇫🇷",  # French
        # Add more languages and their respective flag emojis as needed
    }

    # Get the flag emoji for the detected language
    flag_emoji = language_flags.get(detected_language, "🔍")

    # Make a Streamlit toast
    st.toast(f"Language detected: {flag_emoji}", icon="🔍")
    
    # Get the intermediate steps
    with st.spinner("Getting intermediate thinking steps..."):
        intermediate_steps = query_breaker.break_query(query, context=chat_history)
        intermediate_steps = [str(step) for step in intermediate_steps]
        
    st.toast("Intermediate steps computed!")
    
    print("#################################################")
    print("INTERMEDIATE STEPS LIST:", intermediate_steps)
    
    # For each step we launch the RAG pipeline 
    list_sources = []  # we keep track of the sources used
    step_number = 0
    for step in intermediate_steps:
        print("#################################################")
        print("TREATING STEP NUMBER:", step_number)
        # First we translate the step in a prompt for the rag pipeline
        with st.spinner(f"🔄 Working on step {step_number + 1} out of {len(intermediate_steps)}..."):
            prompt = task_translator.get_prompt(step, context=memory.get_content())
            print("#################################################")
            print("TASK PROMPT:", prompt)
            # Then we call the RAG pipeline on the prompt
        with st.spinner(f"🤖 Answering intermediate query: {prompt}"):
            answer, docs, sources = RAG_answer(prompt, default_config, config={"return_chunks": True})
            sources = [str(source) for source in sources]
            print("#################################################")
            print("INTERMEDIATE ANSWER:", answer)
            # We add the answer to the memory
            memory.ingest_info(f"{{STEP {step_number + 1}: {step} QUERY: {prompt} ANSWER: {answer}}}")
            step_number += 1
        #Add the sources to the list of sources
        list_sources.extend(sources)
    
    #Delete duplicates in the list of sources
    list_sources = list(set(list_sources))
    
    # Formulate the final answer using the memory
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
        
        
        


def entity_description(entity, default_config, config={"stream": False}):
    """
    function that takes an entity from the knowledge graph as input and returns a description of the entity using gathered information from the
    vector search.
    input:
        entity: the entity from the knowledge graph. type: string
        default_config: the default configuration for the vector search. type: dict
        config: the configuration for the vector search. type: dict
    output:
        answer: The answer in str format
    """
    # Merge default_config and config. If a key exists in both, value from config is used.
    merged_config = {**default_config, **config}

    # Configure logging
    logging.basicConfig(
        filename="rag_answer.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    # Log merged config field filters
    logging.info("FINAL CONFIG FILTERS: %s", merged_config["field_filter"])

    # we search the useful docs in the database

    prompt = "Qu'est ce que l'entité suivante: " + entity + " ?"

    useful_docs = query_database_v2(prompt, default_config, config)

    logging.info("NUMBER OF DOCS FROM QUERY DATABASE : %d", len(useful_docs))

    start_time = time.time()
    # We make a str using the document content and the metadata
    list_content = [doc.page_content for doc in useful_docs]

    # we list the metadata of the documents
    list_metadata = [doc.metadata for doc in useful_docs]

    # We create a context using all the documents and \n\n as a separator
    list_context = [
        f"Document {i}: {content} \n\n "
        for i, (metadata, content) in enumerate(
            zip(list_metadata, list_content)
        )
    ]

    str_context = " ".join(list_context)

    
    # we define the prompt
    prompt = f"Extract a short description in one or two sentences of the following entity based on the provided documents. Here is the entity: <entity> {entity} </entity> and the documents : \n\n {str_context}. \n No need to cite the documents name or page number. Return the description without preamble."

    logging.info("TIME TO FORMAT CONTEXT: %f", time.time() - start_time)

    nb_tokens = token_calculation_prompt(prompt)
    logging.info(
        "APPROXIMATE NUMBER OF TOKENS IN THE PROMPT: %d", nb_tokens
    )

    logging.info("PROMPT: %s", prompt)

    # We get the answer using the LLM model
    stream_generator = LLM_answer_v3(
        prompt,
        model_name=merged_config["description_model_name"],
        llm_provider=merged_config["description_llm_provider"],
        stream=False,
        temperature=merged_config["temperature"],
    )
    
    print("ENTITY DESCRIPTION :", stream_generator)

    return stream_generator


def create_full_prompt(list_raw_docs, cot_enabled):
    """Takes the list of raw documents and the cot_enabled boolean and returns a full well formated prompt for the LLM model.

    Args:
        list_raw_docs (_type_): _description_
        cot_enabled (_type_): _description_

    Returns:
        dictionnary: {prompt: str, list_content: list, list_sources: list}
    """

    # We make a str using the document content and the metadata
    list_content = [
        doc.page_content
        for doc in list_raw_docs  # text_preprocessing is not needed here because its already done in the vectorstore creation !!
    ]

    # we list the metadata of the documents
    list_metadata = [doc.metadata for doc in list_raw_docs]
    # We list the sources of the documents
    list_sources = [metadata["source"] for metadata in list_metadata]

    list_context = [
        f"Document_{i}: {content} \n\n "  # Source: {metadata['source']}, Content:  NOT SURE IF WE NEED THIS !!!!
        for i, (metadata, content) in enumerate(
            zip(list_metadata, list_content)
        )
    ]

    str_context = " ".join(list_context)

    if cot_enabled:
        # We define a COT prompt for the LLM model
        prompt = f"Réponds à la question suivante : {query} . Pour y répondre, tu utiliseras les informations contenues dans les documents suivants, tu prendras soin de préciser quel(s) document(s) tu as utilisé pour trouver la réponse (nom du doc et page) et de répondre en français. Explicite le raisonnement par étapes ayant mené à ta réponse avant de la formuler. Ton raisonnement peut te mener à ne pas trouver de réponse. Si tu relèves une incohérence à une étape de ton raisonnement, tu peux t'arrêter et conclure. Voici les documents que tu dois utiliser: \n\n {str_context}"
    else:
        # We define a prompt for the LLM model
        prompt = f"Réponds à la question suivante : {query} . Pour y répondre, tu utiliseras les informations contenues dans les documents suivants: \n\n {str_context}."

    return {
        "prompt": prompt,
        "list_content": list_content,
        "list_sources": list_sources,
    }


if __name__ == "__main__":
    # open the config.yaml file
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # we define the query
    query = "Qui est Guillaume ?"

    # we get the answer
    answer = RAG_answer(
        query,
        default_config=config,
        config={"stream": False, "return_chunks": False},
    )
