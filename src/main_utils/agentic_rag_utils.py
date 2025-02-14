import json
from typing import List

# load environment variables
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.main_utils.generation_utils_v2 import LLM_answer_v3

load_dotenv()


class ReformulatedQuery(BaseModel):
    query: str


class TaskTranslator:
    """
    A class to handle the translation of tasks into prompts for a language model.

    Attributes:
        config (dict): Configuration dictionary containing settings for the translator.
        prompt_language (str): The language to be used for the prompt, derived from the config.

    Methods:
        __init__(config):
            Initializes the TaskTranslator with the given configuration.

        get_prompt(task, context):
            Generates a prompt based on the task and context, and retrieves the response from the language model.
    """
    def __init__(self, config):
        self.config = config
        self.prompt_language = config["prompt_language"]

    def get_prompt(self, task, context):
        template_file = f"prompts/task_translator_{self.prompt_language}.txt"

        with open(template_file, "r", encoding="utf-8") as file:
            template = file.read()

        prompt_template = PromptTemplate.from_template(template)
        self.contextualisation_prompt = prompt_template.format(
            task=task, context=context
        )

        print("TRANSLATOR PROMPT:", self.contextualisation_prompt)
        return LLM_answer_v3(
            self.contextualisation_prompt,
            json_formatting=False,
            model_name=self.config["model_name"],
            llm_provider=self.config["llm_provider"],
            stream=False,
        )


class ChabotMemory:
    """
    Memory Of The RAG Chatbot
    Can store past interactions and provide context to the LLM, or can be used for reasoning
    """

    def __init__(self, config):
        self.chat_history = []  # list of dict {'role': 'user', 'content': 'Qui est Alice ?'}, {'role': 'assistant, ...
        self.model_name = config["model_name"]
        self.llm_provider = config["llm_provider"]
        self.context = ""
        self.last_user_question = ""
        with open(
            "prompts/memory_contextualizer.txt", "r", encoding="utf-8"
        ) as file:
            self.template = file.read()

        self.prompt_template = PromptTemplate.from_template(self.template)
        self.memory_content = ""  # The content of the memory

    def ingest_info(self, text):
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
        return self.prompt_template.format(query=query, context=context)

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
    list_steps: List[str] = Field(
        description="List of strings representing the steps. Example: ['Step 1: Identify the people working on Project A.', 'Step 2: Determine who among them is responsible for maintaining the machines.']"
    )


class QueryBreaker:
    """
    A class used to break down queries into sub-steps using a language model.

    Attributes
    ----------
    model_name : str
        The name of the language model to be used.
    llm_provider : str
        The provider of the language model.
    prompt_language : str
        The language in which the prompt template is written.

    Methods
    -------
    break_query(query, context=None)
        Breaks down the given query into sub-steps using the specified language model.
    """
    def __init__(self, config):
        self.model_name = config["model_name"]
        self.llm_provider = config["llm_provider"]
        self.prompt_language = config["prompt_language"]
        self.temperature = config["temperature"]

    def break_query(self, query, context=None,unitary_actions=[]):
        """
        Breaks down a given query into sub-steps using a language model.

        Args:
            query (str): The query to be broken down.
            context (str, optional): Additional context to be included in the prompt. Defaults to None.

        Returns:
            list or str: A list of sub-steps if the answer is in list format, or a string indicating an issue with the answer format.
        """
        template_file = f"prompts/query_breaker_{self.prompt_language}.txt"

        with open(template_file, "r", encoding="utf-8") as file:
            template = file.read()

        prompt_template = PromptTemplate.from_template(template)
        prompt = prompt_template.format(query=query,unitary_actions=unitary_actions)

        if context:
            prompt = f"{prompt}\nContext: {context}"

        answer = LLM_answer_v3(
            prompt,
            json_formatting=True,
            pydantic_object=SubSteps,
            format_type="list",
            model_name=self.model_name,
            llm_provider=self.llm_provider,
            temperature=self.temperature
        )

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
    # Test du QueryBreaker avec une tâche complexe
    config = {
        "model_name": "gemini-2.0-flash",
        "llm_provider": "google",
        "prompt_language": "fr",
        "temperature": 0.7
    }
    
    query_breaker = QueryBreaker(config)
    test_query = "Analyse les performances commerciales de notre entreprise du dernier trimestre et prépare un rapport détaillé avec des recommandations"
    
    print("Test du QueryBreaker avec la requête:")
    print(f"Query: {test_query}\n")
    
    steps = query_breaker.break_query(test_query)
    print("Résultat de la décomposition en étapes:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

