import json

import yaml
from pydantic import BaseModel

from src.evaluation_utils import get_k_random_chunks
from src.generation_utils import LLM_answer_v3
from src.utils import text_preprocessing


def RAFT_dataset(k, n_distractor, config):
    """
    This function returns k random chunks from the database of documents (the ORACLES documents) and samples
    also n_distractor chunks for each of the k chunks.
    input:
        k: int, the number of oracles documents
        n_distractor: int, the number of distractor documents
        config: dict, the configuration file
    output:
        list_dictionnaries: list of dictionaries of shape {"oracle": "oracle", "distractors": ["distractor1", "distractor2", ...], "question": "question", "answer": "answer"}

    """

    class QuestionAnswerPair(BaseModel):
        question: str
        answer: str

    num_queries = 1

    print("TYPE OF CONFIG: ", type(config))

    list_oracles = get_k_random_chunks(
        k, config
    )  # we get the oracles documents
    # process the oracles documents
    list_oracles = [
        text_preprocessing(oracle.page_content) for oracle in list_oracles
    ]
    list_dictionnaries = []  # to store the oracle and the distractors {oracle: "oracle", distractors: ["distractor1", "distractor2", ...]}
    for k in list_oracles:
        list_distractors = get_k_random_chunks(n_distractor, config)
        # we process the distractors
        list_distractors = [
            text_preprocessing(distractor.page_content)
            for distractor in list_distractors
        ]
        list_dictionnaries.append(
            {"oracle": k, "distractors": list_distractors}
        )  # we store the oracle and the distractors

    # generate a question to be answered for each oracle documents and the associated answer

    for k in list_dictionnaries:
        print("Oracle: ", k["oracle"])
        print("Distractors: ", k["distractors"])

        oracle = k["oracle"]

        num_queries = 1

        prompt = f"""
Generate {num_queries} question and the associated answer based on the subject or theme of this document chunk: <document> {oracle} </document>. The question should be formulated in a way that it could generally apply to other content on the same subject, yet can be specifically answered using the information contained within this chunk. Avoid questions that reference ultra-specific details of the document, such as those tied to exact phrases or paragraphs or explicit references to the text (example: 'in the following text, according to this text, mentionned in the document, etc.'). The questions should cover a range of types (what, who, where, when, why, how) and encourage a deep understanding of the subject presented.
"""

        answer_json = LLM_answer_v3(
            prompt, json_formatting=True, pydantic_object=QuestionAnswerPair
        )

        # parsing the answer
        answer = answer_json["answer"]
        question = answer_json["question"]

        # modify the dictionary to add a key for the question and the answer
        k["question"] = question
        k["answer"] = answer

    return list_dictionnaries


def save_to_json(data, filename):
    """
    This function saves a Python object to a JSON file.
    input:
        data: Python object to be saved
        filename: str, the name of the file
    """
    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    # load the config
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)

    k = 30
    n_distractor = 30
    e = RAFT_dataset(k=k, n_distractor=n_distractor, config=config)
    print("################################################################")
    print("RAFT DATASET: ", e)

    # Save the dataset to a JSON file
    save_to_json(e, "RAFT_dataset.json")
