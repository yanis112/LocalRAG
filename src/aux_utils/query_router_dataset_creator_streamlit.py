import json
import os
import random

import streamlit as st
import yaml
from langchain_qdrant import Qdrant
from pydantic import BaseModel

from src.main_utils.embedding_utils import get_embedding_model

# custom imports
from src.main_utils.generation_utils import LLM_answer_v3
from src.main_utils.retrieval_utils import query_database_v2
from src.main_utils.utils import get_k_random_chunks_qdrant


def generate_query_class():
    classes = ["semantic", "lexical"]
    return random.choice(classes)


def get_similar_chunks_from_target(target_chunk):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    docs = query_database_v2(
        target_chunk,
        default_config=config,
        config={
            "nb_chunks": 10,
            "nb_rerank": 5,
            "return_chunks": True,
            "stream": False,
            "advanced_hybrid_search": False,
            "use_reranker": False,
        },
    )

    print("SIMILAR CHUNKS: ", docs)

    return docs


def get_top_chunk_for_query(query):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    docs = query_database_v2(
        query,
        default_config=config,
        config={
            "nb_chunks": 10,
            "nb_rerank": 1,
            "return_chunks": True,
            "stream": False,
            "advanced_hybrid_search": False,
            "use_reranker": False,
        },
    )

    print("TOP CHUNK (1): ", docs)

    return docs[0]


def generate_example(list_chunks):
    if not list_chunks:
        st.error("No more chunks available.")
        return None

    chunk = random.choice(list_chunks)  # Select a random chunk
    list_chunks.remove(chunk)  # Remove the selected chunk from the list

    class LLMANSWER(BaseModel):
        question: str

    prompt = f"""
    Generate 1 question based on the subject or theme of this document chunk: <document> {chunk} </document>.
    The question should be formulated in a way that it could generally apply to other content on the same subject,
    yet can be specifically answered using the information contained within this chunk. Avoid questions that reference
    ultra-specific details of the document, such as those tied to exact phrases or paragraphs or explicit references to
    the text (example: 'in the following text, according to this text, mentioned in the document, etc.'). The question
    should cover a range of types (what, who, where, when, why, how) and encourage a deep understanding of the subject presented. Answer the question in json format. """

    prompt_2 = f"""
    Generate 1 question based on the subject or theme of this document chunk: <document> {chunk} </document>.
    The question should be formulated in a way that it could generally apply to other content on the same subject,
    and can't be specifically answered using the only information contained within this chunk as it is a general, abstract. Avoid questions that reference
    ultra-specific details of the document, such as those tied to exact phrases or paragraphs or explicit references to
    the text (example: 'in the following text, according to this text, mentioned in the document, etc.'). The question
    should cover a range of types (what, who, where, when, why, how) and encourage a deep understanding of the subject presented, a large vision of the question/subject. Answer the question in json format. """

    # make a random choice between the two prompts
    prompt = random.choice([prompt, prompt_2])

    try:
        question = LLM_answer_v3(
            prompt,
            json_formatting=True,
            pydantic_object=LLMANSWER,
            llm_provider="groq",
            model_name="llama3-70b-8192",
            temperature=0,
        )
        question = question["question"]
        return {"chunk": chunk, "query": question}
    except Exception as e:
        print(f"An error occurred: {e}. Retrying...")
        return None


def save_example(example, dataset_path, label):
    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    key = f"ex{len(data) + 1}"
    query = example["query"]
    data[key] = {"query": query, "label": label}

    with open(dataset_path, "w") as f:
        json.dump(data, f)


st.title("Query Router Dataset Generator")


if st.button("Start"):
    file_path = "src/datasets/query_router_finetuning_dataset_4_claases.json"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({}, f)

    st.session_state["dataset_file_path"] = file_path

    # load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embedding_model = get_embedding_model(model_name=config["embedding_model"])

    qdrant_db = Qdrant.from_existing_collection(
        path=config["persist_directory"],
        embedding=embedding_model,
        collection_name="qdrant_vectorstore",
    )
    list_chunks = get_k_random_chunks_qdrant(
        10000, config=config, qdrant_db=qdrant_db
    )[0]

    st.session_state["list_chunks"] = list_chunks

    if st.session_state.get("example") is None and st.session_state.get(
        "list_chunks"
    ):
        with st.spinner("Generating example..."):
            example = generate_example(st.session_state["list_chunks"])
            if example:
                st.session_state["example"] = example
                print("Type of example: ", type(example))
                st.json(example)
                # st.write(example["query"])


col1, col2, col3, col4 = st.columns(4)
with col1:
    very_semantic = st.button("Very Semantic")

with col2:
    lexical = st.button("Semantic")

with col3:
    semantic = st.button("Lexical")

with col4:
    very_lexical = st.button("Very Lexical")


if semantic:
    save_example(
        st.session_state["example"],
        st.session_state["dataset_file_path"],
        label="semantic",
    )
    st.success("Saved!")
    # Generate a new example
    example = generate_example(st.session_state["list_chunks"])
    print("EXAMPLE: ", example)
    st.session_state["example"] = example
    # st.json(example)
    st.write(example["query"])


if very_semantic:
    save_example(
        st.session_state["example"],
        st.session_state["dataset_file_path"],
        label="very_semantic",
    )
    st.success("Saved!")
    # Generate a new example
    example = generate_example(st.session_state["list_chunks"])
    print("EXAMPLE: ", example)
    st.session_state["example"] = example
    # st.json(example)
    st.write(example["query"])


if very_lexical:
    save_example(
        st.session_state["example"],
        st.session_state["dataset_file_path"],
        label="very_lexical",
    )
    st.success("Saved!")
    # Generate a new example
    example = generate_example(st.session_state["list_chunks"])
    print("EXAMPLE: ", example)
    st.session_state["example"] = example
    # st.json(example)
    st.write(example["query"])


if lexical:
    save_example(
        st.session_state["example"],
        st.session_state["dataset_file_path"],
        label="lexical",
    )
    # Generate a new example
    example = generate_example(st.session_state["list_chunks"])
    print("EXAMPLE: ", example)
    # st.json(example)
    # st.session_state["example"] = example
    st.write(example["query"])
