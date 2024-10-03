import json
from typing import List

import streamlit as st

# load environment variables
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.generation_utils import LLM_answer_v3
from src.knowledge_graph import KnowledgeGraph
from src.retrieval_utils import query_database_v2
from src.utils import text_preprocessing

load_dotenv()




class WorldModel: #DEPRECATED
    """
    The goal of the world model is to define a desciptive text presenting to the LLM
    the structure of the database from a higher, abstract point of view.
    """

    def __init__(self):
        self.content = (
            "You have a database containing data from the company Euranova, specialized it Data and artificial intelligence. \
        The database is structured in different collections containing various information about the company, its employees, its projects, its clients, its products and its services. \
        The collection 'mattermost' contains the messages exchanged between the employees of the company to solve technical issues, questions about development, and to share information. \
        The collection 'gitlab' contains the code repositories of the company and descriptions of the projects, code, tools used and python modules. \
        The collection 'happeo' contains general information about the company, its employees, enboarding process, and the company's values. \
        The collection 'drive"
        )

    def get_embedding(self):
        # prompt for getting the embedding of the word
        prompt = f"Get the embedding of the word '{self.word}' using the model '{self.model_name}'."

        # Use LLM_answer to get the embedding of the word
        embedding = LLM_answer_v3(
            prompt, model_name=self.model_name, llm_provider=self.llm_provider
        )

        return embedding


class ReformulatedQuery(BaseModel):
    query: str


class TaskTranslator:
    """
    Translate a task and a context into a RAG fitting prompt.
    """
    
    def __init__(self, config):
        self.config = config
        self.prompt_language = config["prompt_language"]
        
    def get_prompt(self, task, context):
        """
        Args:
            task (str): The task description.
            context (str): The context information.
        Returns:
            str: The task translated into a RAG fit question without any introduction or explanation.
        """
        
        if self.prompt_language == "en":
            self.contextualisation_prompt = f"""Formulate a question for an information search bot that is clear without needing any given context aside from the intrisic information contained in the question. For example: 

- If the Task is 'Find out where the person studied' and the Context is 'Alice works at Euranova', the reformulated question could be 'Where did Alice study?'

Now, here are the task and context:

- Task: {task} 
- Context: {context}

Respond with the reformulated question only, without any introduction or explanation. """
        else:  # Assuming the prompt_language is 'fr'
            self.contextualisation_prompt = f"""Formulez une question pour un bot de recherche d'informations qui soit claire sans avoir besoin de contexte donné à pars les informations intrinsèques à la question. Par exemple :

- Si la tâche est 'Découvrez où la personne a étudié' et le contexte est 'Alice travaille chez Euranova', la question reformulée pourrait être 'Où Alice a-t-elle étudié ?'

Voici maintenant la tâche et le contexte :

- Tâche : {task} 
- Contexte : {context}

Répondez uniquement avec la question reformulée, sans aucune introduction ou explication. """

        print("TRANSLATOR PROMPT:", self.contextualisation_prompt)
            
        prompt = LLM_answer_v3(self.contextualisation_prompt, json_formatting=False, model_name=self.config['model_name'], llm_provider=self.config['llm_provider'], stream=False, temperature=0)
            
        return prompt


class ChabotMemory:
    """
    Memory Of The RAG Chatbot
    Can store past interactions and provide context to the LLM, or can be used for reasoning
    """
    def __init__(self, config):
        self.chat_history = []  # list of dict {'role': 'user', 'content': 'Qui est Alice ?'}, {'role': 'assistant, ...
        self.model_name = config['model_name']
        self.llm_provider = config['llm_provider']
        self.context = ""
        self.last_user_question = ""
        self.contextualisation_prompt = f"Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is. Here is the latest user question: \n {self.last_user_question} \n \
            And here is the chat history: \n {self.chat_history} ."
        #self.memory_size = config["memory_size"] #Maximum number of tokens stored in the memory
        self.memory_content = "" #The content of the memory

    def ingest_info(self,text):
        """
        Add the text to the memory
        """
        self.memory_content += text
    
    def get_content(self):
        """
        Get the content of the memory
        """
        return self.memory_content
        
    def clear_memory(self):
        """
        Clear the memory
        """
        self.memory_content = ""

    def update_chat_history(self, list_messages):
        """
        input format: list of dict {'role': 'user', 'content: 'Qui est Alice ?'}, {'role': 'assistant', ...
        """

        self.chat_history = list_messages

    def get_prompt(self, query, context):
        """
        Create a prompt for the LLM to contextualise the query
        """
        self.contextualisation_prompt = f"Given a chat history and the latest user question\
        which might reference context in the chat history, formulate a standalone question\
        that expands the user question in a way that it can be understood without the chat history, by adding contextual elements.\
        If the question contains enough contextual elements to be answered return it as is, otherwise add usefull contextual elements. Here is the latest user question to reformulate: \n {query} \n\
        And here is the chat history: \n {context} ."

        return self.contextualisation_prompt

    def contextualise_query(self, query, n_messages=3):
        """
        input: query as string
        output: query as string with the context of the chat history
        """
        # Get the last n messages interectations (an interaction is a pair of dicts..) #we dont want to include the last query in the context
        last_messages = self.chat_history[(-n_messages - 1) * 2 : -1]

        # Create a string with the last 3 messages using json dumps
        context = json.dumps(last_messages)

        # get the prompt
        prompt = self.get_prompt(query, context)

        print("MEMORY PROMPT:", prompt)

        # Get the answer from the LLM and parse it
        new_query = LLM_answer_v3(
            prompt=prompt,
            model_name=self.model_name,
            llm_provider=self.llm_provider,
            stream=False,
            json_formatting=True,
            pydantic_object=ReformulatedQuery,
        )

        return new_query


class SummaryMemory:
    def __init__(self, model_name, llm_provider="ollama"):
        self.current_summary = "empty summary"
        self.model_name = model_name
        self.new_user_message = None
        self.new_bot_message = None
        self.llm_provider = llm_provider

    def update_summary(self, new_user_message, new_bot_message):
        self.new_user_message = new_user_message
        self.new_bot_message = new_bot_message

        self.prompt = f"Using the current dialogue summary and the new user_message and bot_message, update the current summary (even if empty) with the fresh information , keep all previous information that is necessary for answering the user's questions. \n \
        Current summary: \n{self.current_summary} \n\n New user message: \n{self.new_user_message} \n\n New bot message: \n{self.new_bot_message} \n\n Now update the summary with the new information: \n New summary: \n "

        # Get the answer from the LLM
        new_summary = LLM_answer_v3(
            prompt=self.prompt,
            model_name=self.model_name,
            stream=False,
            llm_provider=self.llm_provider,
        )

        self.current_summary = new_summary

    def get_summary(self):
        return self.current_summary


class SubSteps(BaseModel):
    list_steps: List[str] = Field(description="List of strings representing the steps. Example: ['Step 1: Identify the people working on Project A.', 'Step 2: Determine who among them is responsible for maintaining the machines.']")


class QueryBreaker:
    def __init__(self, config):
        """
        Initializes an instance of the QueryBreaker class.

        Args:
            config (dict): A dictionary containing the configuration parameters.
                - model_name (str): The name of the model.
                - llm_provider (str): The provider of the LLM.
                - prompt_language (str): The language of the prompt ('en' for English, 'fr' for French).

        Attributes:
            model_name (str): The name of the model.
            llm_provider (str): The provider of the LLM.
            prompt_language (str): The language in which prompts will be generated.
            query (None): The query attribute is initially set to None.
        """
        self.model_name = config["model_name"]
        self.llm_provider = config["llm_provider"]
        self.prompt_language = config["prompt_language"]
        self.query = None

    def break_query(self, query, context=None):
        """
        Breaks down a complex query into 2 or 3 simpler, independent steps that can be answered sequentially.
        Args:
            query (str): The complex query to be broken down.
            context (str, optional): The context in which the query is asked. Defaults to None.
        Returns:
            list or str: A list of 2 or 3 simpler steps that, when answered sequentially, will comprehensively address the original complex query. Each step is clear and specific. If the answer format is a list, it returns the list of steps. If the answer format is a dictionary, it returns the value associated with the "steps" key. If the answer format is neither a list nor a dictionary, it returns "ISSUE WITH THE ANSWER FORMAT".
        
        """
        self.query = query

        if self.prompt_language == "en":
            prompt = f"""
            You are an AI designed to break down complex queries into 2 or 3 simpler, independent steps that can be answered sequentially to address the query. For the following complex query: "{self.query}", generate a list of 2 or 3 steps that, when answered in order, will fully address the original query. Each step should be clear and specific.

Example query: "Who is responsible for maintaining the machines used in Project A?"
Example steps:
1. Identify the people working on Project A.
2. Determine who among them is responsible for maintaining the machines.

Now, please generate steps for the following complex query: "{self.query}".
            """
            if context:  
                prompt = f"""You are an AI designed to break down complex queries into 2 or 3 simpler, independent steps that can be answered sequentially to address the query. For the following complex query: "{self.query}", generate a list of 2 or 3 steps that, when answered in order, will fully address the original query. Please take the provided context into account when breaking down the query, the context consists of the previous questions of the user and the answers given by the chatbot, so that you can deduce his intentions and the context of his question.

Context: {context}

Example query: "Who is responsible for maintaining the machines used in Project A?"
Example steps:
1. Identify the people working on Project A.
2. Determine who among them is responsible for maintaining the machines.

                """
        else:  # Assuming the prompt_language is 'fr'
            prompt = f"""
            Vous êtes une IA conçue pour décomposer des requêtes complexes en 2 ou 3 étapes plus simples et indépendantes qui peuvent être répondues séquentiellement pour traiter la requête. Pour la requête complexe suivante : "{self.query}", générez une liste de 2 ou 3 étapes qui, une fois répondues dans l'ordre, traiteront complètement la requête originale. Chaque étape doit être claire et spécifique.

Exemple de requête : "Qui est responsable de l'entretien des machines utilisées dans le projet A ?"
Exemple d'étapes :
1. Identifier les personnes travaillant sur le projet A.
2. Déterminer qui parmi eux est responsable de l'entretien des machines.

Maintenant, veuillez générer des étapes pour la requête complexe suivante : "{self.query}".
            """
            if context:  
                prompt = f"""Vous êtes une IA conçue pour décomposer des requêtes complexes en 2 ou 3 étapes plus simples et indépendantes qui peuvent être répondues séquentiellement pour traiter la requête. Pour la requête complexe suivante : "{self.query}", générez une liste de 2 ou 3 étapes qui, une fois répondues dans l'ordre, traiteront complètement la requête originale. Veuillez prendre en compte le contexte fourni lors de la décomposition de la requête.

Contexte : {context}

Exemple de requête : "Qui est responsable de l'entretien des machines utilisées dans le projet A ?"
Exemple d'étapes :
1. Identifier les personnes travaillant sur le projet A.
2. Déterminer qui parmi eux est responsable de l'entretien des machines.

                """

        # Use LLM_answer to get a list of steps
        answer = LLM_answer_v3(
            prompt,
            json_formatting=True,
            pydantic_object=SubSteps,
            format_type="list",
            model_name=self.model_name,
            llm_provider=self.llm_provider,
            temperature=0,
        )
        print("RAW ANSWER:", answer)

        if isinstance(answer, list):
            return answer
        elif isinstance(answer, dict):
            return answer["list_steps"]
        else:
            return "ISSUE WITH THE ANSWER FORMAT !"




def answer_relevancy_classifier(query, answer): #DEPRECATED
    """
    function that takes the query and the answer as input and returns the relevancy of the answer
    input:
        query: the query to search in the database. type: string
        answer: the answer to the query. type: string
    output:
        relevancy: True il the answer is relevant, False otherwise
    """

    prompt = f"Here is a question: {query} and here is an answer obtained from a RAG pipeline and formullated by a LLM: {answer}. Does the answer answer fully the question or is lacking key elements (no explicit mention, could not answer, ect ...) ? Classify the answer in those three categories: 'fully answered' if all the key and precise details are provided, 'partially answered' in a general answer is given but not detailled enough, 'not answered' if the answer contains i coulndt answer or i dont know."

    # define pydantic object
    class Relevancy(BaseModel):
        relevancy: str

    answer = LLM_answer_v3(
        prompt,
        json_formatting=True,
        pydantic_object=Relevancy,
        model_name="llama3",
        llm_provider="ollama",
    )

    # parse the answer
    relevancy = answer["relevancy"]

    return relevancy


def answer_from_knowledge_graph(query, kg): #DEPRECATED
    """
    function that takes the knowledge graph as input and returns the answer
    input:
        kg: the knowledge graph. type: KnowledgeGraph
    output:
        answer: The answer in str format
    """
    # we get the answer from the knowledge graph
    dict_graph = kg.graph_to_dict()

    print("DICT GRAPH:", dict_graph)

    # prompt for answerinf the question based on the knowledge graph
    prompt = f"Réponds à la question suivante : {query} en utilisant les informations contenues dans le knowledge graph suivant, utilise tout les infos pertinantes disponibles pour répondre à la question: <knowledge_graph> {str(dict_graph)} </knowledge_graph>."

    # get the answer from the LLM model

    answer = LLM_answer_v3(prompt)

    return answer


class AggregatorReranker:
    def __init__(
        self, reranker_name="jinaai/jina-reranker-v2-base-multilingual"
    ):
        self.reranker_name = reranker_name

    def get_scores(self, query, list_documents):
        """
        function that takes the query and the documents as input and returns the scores of the documents
        input:
            query: the query to search in the database. type: string
            documents: the list of langchain documents objects to rerank. type: list
        output:
            scores: The scores of the documents in a list
        """
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        model.to("cuda")  # or 'cpu' if no GPU is available
        model.eval()

        raw_contents = [doc.page_content for doc in list_documents]

        # Construct pairs of query and document
        document_pairs = [[query, doc] for doc in raw_contents]

        # Compute scores
        scores = model.compute_score(document_pairs, max_length=1024)

        return list_documents, scores

    def rerank_top_k(self, query, list_documents, k=5):
        # Get the scores
        list_documents, scores = self.get_scores(query, list_documents)

        # return the top k best documents, not the scores
        top_k = [doc for doc in list_documents[:k]]
        return top_k


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
    model_name = "llama3-70b-8192"
    sm = SummaryMemory(model_name=model_name)
    user_message = "Comment se connecter à Vidar en ssh ?"
    bot_message = "Bonjour ! \
    Pour vous connecter à Vidar en ssh, voici les étapes à suivre : \
    1. Oubliez Tahiti-Bob, votre username c'est Aziz Jedidi ! \
    2. Essayez la commande `ssh aziz.jedidi@vidar.enx` (ou simplement `ssh vidar` si vous ne précisez pas de user). \
    3. Si cela ne fonctionne pas, assurez-vous d'avoir ajouté votre SSH key dans le console Jumpcloud. \
    Notez que si vous arrivez à vous connecter sur la machine physique avec votre user et mot de passe, il est possible que vous n'ayez pas de timeout sur Vidar. \
    Si vous avez des problèmes pour se connecter, essayez de scanner le réseau pour trouver la nouvelle adresse IP de Vidar (cf. Document 2). \
    Et si cela ne marche toujours pas, vous pouvez essayer d'ajouter les lignes `Host * ForwardX11 yes ForwardX11Trusted yes` dans votre fichier `~/.ssh/config` (cf. Document 3). Cela devrait vous aider à résoudre les problèmes de connexion.\
    J'espère que cela vous aidera à vous connecter à Vidar en ssh !"

    sm.update_summary(user_message, bot_message)

    print("New Summary:", sm.get_summary())

    user_message = (
        "Comment se connecter n'importe qu'elle machine distante en ssh ?"
    )
    bot_message = (
        "Pour vous connecter à n'importe quelle machine distante en ssh, il est nécéssaire de d'abord créer \
        une clé ssh et de l'ajouter à la machine distante. Ensuite, vous pouvez vous connecter en utilisant la commande `ssh user@machine`, \
        il faudra également installer les extensions vscode pour faciliter la connexion. Pour plus d'informations, vous pouvez consulter le \
        document 4."
    )

    sm.update_summary(user_message, bot_message)

    print("###############################################")
    print("New Summary:", sm.get_summary())
