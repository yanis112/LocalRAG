import json
from typing import List

# load environment variables
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.main_utils.generation_utils import LLM_answer_v3

load_dotenv()


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



if __name__ == "__main__":
    pass