import os
import random


import streamlit as st
from retrieval_utils import find_loader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Streamlit interface
st.title("Document Chunker")
test_docs_path = "./test_docs"
doc_names = os.listdir(test_docs_path)
selected_doc_name = st.selectbox("Select a document", doc_names)

# Function to load the document
def load_document(name, path):
    full_path = os.path.join(path, name)

    if name.split(".")[-1] in ["pdf", "docx", "xlsx", "html", "pptx", "md"]:
        print(f"Ce document n'a pas encore été traité, Nom du document: {name}")

        loader = find_loader(name, full_path)
        print("Approprite loader found:", loader)

        doc = loader.load()
        return doc


# un curseur entre 0 et 100 pour définir le seuil de coupure
threshold = st.slider("Threshold", 0, 100, 70)


# Function to chunk the document
def chunk_document(doc):
    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    sementic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )

    chunks = sementic_splitter.create_documents([doc[0].page_content])
    return chunks



# Function to generate a random color
def get_random_color():
    r = lambda: random.randint(0, 255)
    return "#%02X%02X%02X" % (r(), r(), r())


# Function to display chunks in Streamlit


# Function to display chunks in Streamlit
def display_chunks(chunks):
    html_string = ""
    for i, chunk in enumerate(chunks):
        color = get_random_color()
        html_string += f"<p style='color: {color};'>{chunk}</p>"
    st.markdown(html_string, unsafe_allow_html=True)
    st.write("Nombre de chunks:", len(chunks))






doc = load_document(selected_doc_name, test_docs_path)
chunks = chunk_document(doc)
display_chunks(chunks)
