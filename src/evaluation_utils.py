import json
from functools import lru_cache
from typing import List

import numpy as np
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic.main import create_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

# custom imports
from src.generation_utils import LLM_answer_v3
from src.utils import (
    get_k_random_chunks,
    text_preprocessing,
)

load_dotenv()


def trustfullness(query, answer): #DEPRECATED
    """
    function that takes the query and the answer and returns the trustfullness of the answer
    input:
        query: the query to search in the database. type: string
        answer: the answer to evaluate. type: string
    output: a qualification of the trustfullness of the answer between ["Trustfull","Imprecise","Not trustfull"]
    """

    prompt = (
        "Given the question: "
        + query
        + " and the answer: "
        + answer
        + " how would you qualify the trustfullness of the answer ? Return one of the following: Trustfull if the answer precisely and factually answers the question, Imprecise if the answer is not precise or lacking key information, Not trustfull if the answer is incorrect or misleading."
    )

    # We get the llm answer in json format
    answer_json = LLM_answer_v3(prompt, json_formatting=True)

    return answer_json


def trustfullness_score(query, answer):
    """
    function that takes the query and the answer and returns the trustfullness score of the answer
    input:
        query: the query to search in the database. type: string
        answer: the answer to evaluate. type: string
    output: the trustfullness score of the answer. type: int between 0 and 5
    """

    prompt = f"""
        You will be given a user_question and system_answer couple.
        Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.

        Here is the scale you should use to rate the system_answer:
        1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
        2: The system_answer is mostly not helpful: misses some key aspects of the question, doubts are expressed
        3: The system_answer is mostly helpful: provides support, but still could be improved
        4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question
        5: The system_answer is perfect: not only addresses all the concerns raised in the question but also provides additional information that could be helpful to the user

        Now here are the question and answer.

        <question> {query} </question>
        <answer> {answer} </answer>

        Please provide uniquely your rating and in json format.
    """

    # We get the llm answer in json format
    class Trust(BaseModel):
        score: str

    answer_json = LLM_answer_v3(
        prompt, json_formatting=True, pydantic_object=Trust
    )

    print("ANSWER json:", answer_json)

    # extract the score from the json
    score = int(answer_json["score"])

    return score


def similarity_score(question, list_retrieved_docs):
    """
    Function that calculates the similarity score between a question and a list of retrieved documents
    input:
        question: the question to evaluate. type: str
        list_retrieved_docs: the list of retrieved documents to evaluate. type: list
    output:
        The similarity score between the question and the list of retrieved documents. type: float
    """

    import requests

    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query(
        {
            "inputs": {
                "source_sentence": question,
                "sentences": list_retrieved_docs,
            },
        }
    )

    return output


from transformers import AutoModelForSequenceClassification


@lru_cache(maxsize=None)
def load_hallucination_model():
    """
    Function that loads the hallucination evaluation model
    output:
        The hallucination evaluation model. type: CrossEncoder
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", trust_remote_code=True
    )
    return model


def compute_hallucination_scores(answer, list_docs):
    """
    Computes the hallucination score for a given answer and a list of documents.

    Parameters:
    answer (str): The answer to evaluate for hallucination.
    docs (list): A list of documents to compare the answer against.

    Returns:
    float: The hallucination score, which represents the probability of hallucination.
    """

    model = load_hallucination_model()
    batch = [(k, answer) for k in list_docs]

    # predict
    scores = model.predict(batch)

    hallu_probs = [1 - float(score) for score in scores]
    return hallu_probs


def mean_hallucination_score(answer, list_docs):
    """
    Computes the mean hallucination score for a given answer and a list of documents.

    Parameters:
    answer (str): The answer to evaluate for hallucination.
    docs (list): A list of documents to compare the answer against.

    Returns:
    float: The mean hallucination score, which represents the probability of hallucination.
    """
    hallu_probs = compute_hallucination_scores(answer, list_docs)
    return np.mean(hallu_probs)


@lru_cache(maxsize=None)
def load_embedding_model():
    """
    Function that loads the embedding model
    output:
        The embedding model. type: SentenceTransformer
    """
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model


def answer_relevancy_score_v1(question, answer, num_artificial_questions=5):
    """Function that calculates the answer relevancy score between a question and an answer
    input:
        question: the original question. type: str
        answer: the answer to evaluate. type: str
        num_artificial_questions: the number of artificial questions to generate. type: int
    output:
        The answer relevancy score between the question and the answer. type: float
    """

    model = load_embedding_model()
    # SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Generate artificial questions
    prompt = f"""
    Generate {num_artificial_questions} questions based on the subject or theme of this document chunk: <document> {answer} </document>. The questions should be formulated in a way that they could generally apply to other content on the same subject, yet can be specifically answered using the information contained within this chunk. Avoid questions that reference ultra-specific details of the document, such as those tied to exact phrases or paragraphs or explicit references to the text (example: 'in the following text, according to this text, mentionned in the document, etc.'). The questions should cover a range of types (what, who, where, when, why, how) and encourage a deep understanding of the subject presented.
    """

    # load the yaml config file
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    artificial_questions = LLM_answer_v3(
        prompt,
        json_formatting=True,
        pydantic_object=create_query_model(num_artificial_questions),
        llm_provider=config["answer_relevancy_provider"],
        model_name=config["answer_relevancy_llm"],
        stream=False,
        temperature=1,
    )

    # print("Artificial questions:", artificial_questions)

    # Calculate the similarity score between the original question and the artificial questions
    question_embedding = model.encode([question], show_progress_bar=False)
    artificial_questions_embeddings = model.encode(
        list(artificial_questions.values()), show_progress_bar=False
    )
    similarity_scores = cosine_similarity(
        question_embedding, artificial_questions_embeddings
    )[0]

    # Return the average similarity score
    return np.mean(similarity_scores)

from src.knowledge_graph import EntityExtractor

def entity_recall(target_chunk, docs):
    """
    Calculate the recall of entities in the answer compared to the entities in the given documents.
    
    Args:
        target_chunk (str): The answer containing entities.
        docs (list): A list of documents containing entities.
        
    Returns:
        float: The recall value, which is the ratio of the number of common entities between the answer and the documents
               to the total number of entities in the documents. Returns 1.0 if there are no entities in the documents.
    """
    
    # Load the configuration file
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    from src.knowledge_graph import load_entity_extractor
    # Initialize the entity extractor
    entity_extractor = load_entity_extractor("config/config.yaml")

    # Extract entities from the target chunk
    answer_entities = entity_extractor.extract_entities(target_chunk)
    answer_entities = set([k["text"] for k in answer_entities])
    #print("Answer entities:",answer_entities)
    
    # Extract entities from the documents
    doc_entities = set()
    for doc in docs:
        entities = entity_extractor.extract_entities(doc)
        list_founded_entities = [k["text"] for k in entities]
        #print("Entities in the document:",list_founded_entities)
        doc_entities.update(list_founded_entities)
    
    # Calculate the intersection of entities
    intersection = answer_entities.intersection(doc_entities)
    
    #print("Intersection:",intersection)

    # Calculate the recall
    if len(doc_entities) == 0:
        return 1.0  # Return 1.0 if there are no entities in the documents
    
    recall = len(intersection) / len(answer_entities) if len(answer_entities) > 0 else 1.0

    return recall

def faithfulness_score(answer, context_list):
    """
    fucntion that calculates the faithfulness score of the answer by compraing the claims made in the answer and noting (0 or 1) if the claims are supported by the context
    input:
        answer: the answer to evaluate. type: str
        context_list: the list of contexts to evaluate. type: list
    """

    complete_context = " ".join(context_list)

    # extract the list of claims from the answer
    claims = claims_extractor(answer)

    # initialize the score
    score = 0
    # iterate over the claims
    for claim in claims:
        # get the faithfullness of the claim
        faithfullness = few_shot_faithfull_judge(claim, complete_context)
        # if the claim is supported by the context we add 1 to the score
        if faithfullness == 1:
            score += 1

    return score / len(claims)


def claims_extractor(answer): #DEPRECATED
    """
    function that extracts a certain number claims from an answer
    input:
        answer: the answer to extract claims from. type: str
    output:
        list of claims extracted from the answer. type: list
    """

    prompt = f"""
    Given the following answer to a query, extract a list of the claims made in the answer:
    Answer: {answer}
    """

    # We get the llm answer in json format
    class Claims(BaseModel):
        claims: List[
            str
        ]  # = Field(default_factory=list,description="List of claims extracted from the answer")

    answer_json = LLM_answer_v3(
        prompt, json_formatting=True, pydantic_object=Claims
    )

    # extract the claims from the json
    # claims=answer_json["claims"]
    # print("CLAIMS:",claims)

    # return claims


def few_shot_faithfull_judge(claim, context): #DEPRECATED
    """
    function thats takes a claim and a context and returns if the claim is supported by the context (0 if not 1 if yes)
    input:
        claim: the claim to evaluate. type: str
        context: the context to evaluate. type: str
    output:
        The verdict of the claim based on the context. type: int
    """

    prompt = f"""Your task is to judge the faithfulness of a statements based on a given context. You must return verdict as 1 if the statement can be verified based on the context or 0 if the statement can not be verified based on the context.

    Here are some examples to help you understand the task:

    ## Example 1
    Claim: "France is a country in North America."
    Context: "France is a country located in Western Europe, bordered by several countries including Germany, Belgium, and Spain."
    Verdict: 0

    ## Example 2
    Claim: "The human nose can detect over 1 trillion different scents."
    Context: "The human sense of smell is closely tied to the brain's emotional centers, and researchers estimate that we can detect an astonishing number of different odors."
    Verdict: 1

    ## Example 3
    Claim: "The highest mountain in the solar system is Olympus Mons on Mars."
    Context: "Olympus Mons, located on Mars, is the largest volcano in the solar system and has a height of over 27 km above the Martian surface."
    Verdict: 1

    ## Example 4
    Claim: "The shortest war in history was between France and Britain."
    Context: "The Anglo-Zanzibar War was a military conflict fought between the United Kingdom and Zanzibar on August 27, 1896, and lasted only 38 minutes."
    Verdict: 0

   
    ## Task
    Now, judge the faithfulness of the following claim based on the given context:
    Claim: {claim}
    Context: {context}
    Verdict: """

    # We get the llm answer in json format
    class Trust(BaseModel):
        verdict: int  # = Field(description="0 if the statement can not be verified based on the context, 1 if the statement can be verified based on the context")

    answer_json = LLM_answer_v3(
        prompt, json_formatting=True, pydantic_object=Trust
    )

    print("ANSWER json:", answer_json)

    # extract the score from the json
    verdict = answer_json["verdict"]

    print("VERDICT:", verdict)

    return verdict


def ROUGE_score(question, answer): #DEPRECATED
    """
    Function that calculates the ROUGE score between a question and an answer
    input:
        question: the question to evaluate. type: str
        answer: the answer to evaluate. type: str
    output:
        The ROUGE score between the question and the answer. type: dict
    """
    rouge = Rouge()
    scores = rouge.get_scores(answer, question, avg=True)
    return scores

def create_query_model(k):
    fields = {f"query{i}": (str,...) for i in range(1, k + 1)}
    return create_model("QueryList", **fields)


def init_evaluation_dataset(
    num_exemples,
    num_queries,
    embedding_model="All-MiniLM-L6-v2",
    dataset_path="evaluation_dataset.json",
):
    """
    Function that creates an evaluation dataset of num_exemples examples by sampling num_exemples random chunks from the chroma database and then generation artificial questions for each chunk
    input:
        num_exemples: the number of examples to generate. type: int
    output:
        dictionarry containing the examples. type: dict
    """

    # sample k chunks
    chunks = get_k_random_chunks(num_exemples, embedding_model=embedding_model)[
        0
    ]  # we only need the chunks (1 would be the metadatas)
    print("Random chunks from the chroma database found !")

    # extract the content of the chunks
    contents = [text_preprocessing(chunk.page_content) for chunk in chunks]
    print("Contents of the chunks extracted !")

    # generate artificial questions
    # prompt=f"Generate a list of {num_queries} questions for the following document chunk in a way that the questions can be answered by the content of the chunk:\n"

    # initialize examples dict
    examples_dict = {}

    # initialize the query model
    QueryList = create_query_model(num_queries)

    for i, content in tqdm(enumerate(contents), total=len(contents)):
        prompt = f"""
Generate {num_queries} questions based on the subject or theme of this document chunk: <document> {content} </document>. The questions should be formulated in a way that they could generally apply to other content on the same subject, yet can be specifically answered using the information contained within this chunk. Avoid questions that reference ultra-specific details of the document, such as those tied to exact phrases or paragraphs or explicit references to the text (example: 'in the following text, according to this text, mentionned in the document, etc.'). The questions should cover a range of types (what, who, where, when, why, how) and encourage a deep understanding of the subject presented.
"""
        # get an answer from the llm model
        answer = LLM_answer_v3(
            prompt, json_formatting=True, pydantic_object=QueryList
        )
        print("ANSWER json:", answer)
        # the dictionnary obtained is {query1:'query1',query2:'query2'...}
        # Add the chunk and its questions to the examples dictionary
        examples_dict[f"ex{i+1}"] = {"chunk": content, "questions": answer}

    # Save the examples dictionary as a JSON file
    with open(dataset_path, "w") as f:
        json.dump(examples_dict, f)


def load_json(filename):
    """
    Function that loads a JSON file and returns its content.
    input:
        filename: the name of the file to load. type: str
    output:
        The content of the JSON file. type: dict
    """
    with open(filename, "r") as f:
        data = json.load(f)
    return data
