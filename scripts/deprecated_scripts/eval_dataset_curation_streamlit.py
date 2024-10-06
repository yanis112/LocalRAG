import json
import os

import streamlit as st
import yaml
from pydantic import BaseModel

# custom imports
from src.evaluation_utils import (
    create_query_model,
    #get_k_random_chunks,
    #get_k_random_chunks_qdrant,
    text_preprocessing,
)
from src.utils import get_k_random_chunks, get_k_random_chunks_qdrant

from src.generation_utils import LLM_answer_v3
from src.embedding_model import get_embedding_model
from langchain_qdrant import Qdrant

def generate_example(list_chunks=None,config=None):
    chunk = list_chunks.pop()
    content = text_preprocessing(chunk)

    # Check if the chunk has less than 20 words
    while len(content.split()) < 40:
        if list_chunks:  # Check if there are more chunks to pop
            chunk = list_chunks.pop()
            content = text_preprocessing(chunk)
        else:
            return None  # Return None if all chunks have less than 20 words
        
    
    print("CONTENT:", content)
    
    #we create a pydantic object for structured llm output

    class QueryObject(BaseModel):
        query: str


    #QueryList = create_query_model(1)

    #print("QueryList Object:", QueryList)

    prompt = f""" Generate 1 question based on the subject or theme of the given document chunk.
The question should be formulated in a way that it could generally apply to other content on the same subject,
yet can be specifically answered using the information contained within this chunk. Avoid questions that reference
ultra-specific details of the document, such as those tied to exact phrases or paragraphs or explicit references to
the text (example: 'in the following text, according to this text, mentioned in the document,in this document, etc...'). Ensure the question includes enough context to be understood without needing additional information from the document. The question should cover a range of types (what, who, where, when, why, how) and encourage a deep understanding of the subject presented. Answer the question in json format. Here is an example:
<document>Trump received a Bachelor of Science in economics from the University of Pennsylvania in 1968. His father named him president of his real estate business in 1971. 
Trump renamed it the Trump Organization and reoriented the company toward building and renovating skyscrapers, hotels, casinos, and golf courses.</document> OUTPUT: 'What did Trump study at the University of Pennsylvania?' 
Now here is the document you need to generate a question for:
<document> {content} </document>. Answer without preamble.
"""
    #print("PROMPT:", prompt)
    answer = LLM_answer_v3(prompt, json_formatting=True, pydantic_object=QueryObject, model_name=config['model_name'],llm_provider=config['llm_provider'],stream=False,temperature=0) #Very important to set stream to False !
    print("ANSWER:", answer)
    
    return {"chunk": content, "questions": answer['query']}


def save_example(example, dataset_path):
    """
    Save an example to a dataset file. / create the dataset file if it does not exist.

    Args:
        example (Any): The example to be saved.
        dataset_path (str): The path to the dataset file.

    Returns:
        None
    """
    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
        key = int(list(data.keys())[-1]) + 1
    except:
        data = {}
        with open(dataset_path, "w") as f:
            json.dump(data, f)
        key = 1
    data[key] = example
    with open(dataset_path, "w") as f:
        json.dump(data, f)
        
    return key
        
if __name__ == '__main__':
    st.title("Dataset Generator")

    # A cursor to indicate the number of chunks we want to retrieve between 1 and 1000 with a step of 1
    n_chunks = st.slider("Number of chunks", 1, 1000, 1)

    # Load the config file
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
        
    #find the vectorstore provider
    vectorstore_provider = config["vectordb_provider"]

    if st.button("Start"):
        # based on the config we know the perist_directory of the database so we create/open the evaluation file in this directory
        # the name of the file will be "name_of_the_folder + evaluation_dataset.json"
        persist_directory = config["persist_directory"]
        # create the file name based on the persist_directory
        file_name = persist_directory.split("/")[-1] + "_evaluation_dataset.json"
        print("Associated Evaluation Dataset:", file_name)
        # we create its path by joining the persist_directory with the file_name
        file_path = os.path.join(persist_directory, file_name)
        print("PATH OF THE DATASET FILE:", file_path)

        # if the file does not exist we create it
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)
                print("DATASET FILE CREATED")
        else:
            print("DATASET FILE ALREADY EXISTS")

        # we add this path as a session state
        st.session_state["dataset_file_path"] = file_path

        if vectorstore_provider == "Chroma":
            list_chunks = get_k_random_chunks(n_chunks, config=config)[
            0
        ]  # [0] because we only want the list of chunks not the metadata
        
        elif vectorstore_provider == "Qdrant":
            embedding_model = get_embedding_model(model_name=config["embedding_model"])
                
            qdrant_db= Qdrant.from_existing_collection(
                    path=config["persist_directory"],
                    embedding=embedding_model,
                    collection_name="qdrant_vectorstore",
                )
            list_chunks = get_k_random_chunks_qdrant(n_chunks, config=config, qdrant_db=qdrant_db)[0]
            
            
            
        st.session_state["list_chunks"] = list_chunks
        with st.spinner("Generating example..."):
            exemple = generate_example(list_chunks=st.session_state["list_chunks"],config=config)
            st.session_state["example"] = exemple
            st.json(exemple)

    # Create 2 columns to display the buttons
    col1, col2 = st.columns(2)


    with col1:
        save = st.button("Good ✅")

    with col2:
        bad = st.button("Bad ❌")

    if save:
        number= save_example(
            st.session_state["example"], st.session_state["dataset_file_path"]
        )  # save the example in the dataset file for evaluation
        # save_example(st.session_state["example"], "good_chunks.json") #save the example in the dataset file for good chunks
        st.success("Saved !")
        with st.spinner("Generating example..."):
            exemple = generate_example(list_chunks=st.session_state["list_chunks"],config=config)
            st.session_state["example"] = exemple
            st.json(exemple)

    if bad:  # we dont save the exemple and we generate a new one
        exemple = generate_example(list_chunks=st.session_state["list_chunks"],config=config)
        st.session_state["example"] = exemple
        st.json(exemple)

