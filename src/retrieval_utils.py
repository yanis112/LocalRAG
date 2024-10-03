import json
import logging
import os
import shutil
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from multiprocessing import Manager, Pool
from typing import Union

import nltk
import streamlit as st
import torch
import yaml
from dotenv import load_dotenv
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_community.document_loaders import (
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.retrievers import (
    QdrantSparseVectorRetriever,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import (
    FastEmbedSparse,
    Qdrant,
    QdrantVectorStore,
    RetrievalMode,
)
from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from tqdm import tqdm
from transformers import logging as transformers_logging

# custom imports
from src.custom_langchain_componants import *  # noqa:
from src.embedding_model import get_embedding_model, get_sparse_embedding_model
from src.LLM import CustomChatModel
from src.query_routing_utils import QueryRouter
from src.utils import (
    StructuredExcelLoader,
    get_k_random_chunks,
    get_vram_logging,
    keywords_extraction,
    log_execution_time,
    text_preprocessing,
)

# Load the environment variables (API keys, etc...)
load_dotenv()

# Download the necessary nltk resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("omw-1.4")
nltk.download("treebank")
nltk.download("dependency_treebank")


# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="query_database.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf._reader")

# load the config file and embedding model
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# put the embedding model in cache
embeddings = get_embedding_model(model_name=config["embedding_model"])


@log_execution_time
def directory_to_vectorstore(default_config, config={}):
    """
    Converts documents in a directory to langchain Documents objects chunks and adds them to the vectorstore.

    Args:
        path (str): The path to the directory containing the documents.
        chunk_size (int, optional): The size of each chunk in number of characters. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks in number of characters. Defaults to 5.
        embedding_model (str, optional): The name of the Hugging Face embedding model to use. Defaults to "all-mpnet-base-v2".
        splitting_method (str, optional): The method to use for splitting the documents. Can be "recursive" or "semantic". Defaults to "recursive".
        persist_directory (str, optional): The directory to persist the vectorstore. Defaults to './chroma/'.
        process_log_file (str, optional): The name of the log file to store the processed documents. Defaults to 'processed_docs.log'.

    Returns:
        list: A list of langchain document chunks representing the documents in the directory

    Action:
        Adds the documents in the directory to the vectorstore.
    """

    # Merge default_config and provided config. Values in config will override those in default_config.
    config = {**default_config, **config}
    persist_directory = config["persist_directory"]
    process_log_file = config["process_log_file"]
    clone_database = config["clone_database"]
    embedding_model = config["embedding_model"]
    semantic_threshold = config["semantic_threshold"]
    clone_persist = config["clone_persist"]
    clone_embedding_model = config["clone_embedding_model"]
    path = config["path"]
    vectordb_provider = config["vectordb_provider"]
    build_knowledge_graph = config["build_knowledge_graph"]

    # Configure logging for the process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="vector_store_building.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s",
    )

    if not os.path.exists(persist_directory):
        logging.info("Persist directory does not exist!, creating it ...")
        os.makedirs(persist_directory)
        logging.info("Persist directory created successfully!")

    # Get the log file path
    log_file_path = get_log_file_path(persist_directory, process_log_file)

    # Save the config file that was used to create the vectorstore in the persist directory
    with open(
        os.path.join(persist_directory, "building_config.yaml"), "w"
    ) as file:
        yaml.dump(config, file)

    # Load the processed documents
    processed_docs = load_processed_docs(log_file_path)

    list_documents = []
    total_chunks = []

    # In case we want to clone an existing database
    if not clone_database:
        # load all the documents in the directory
        start_time = time.time()
        list_documents = process_documents(
            path=path,
            log_file_path=log_file_path,
            processed_docs=processed_docs,
        )

        end_time = time.time()
        logging.info(
            f"Time taken to process (loading + chunking) the documents: {end_time-start_time} seconds"
        )

        start_time = time.time()
        # split documents into chunks and remove duplicates
        total_chunks = filter_and_split_chunks(
            list_documents=list_documents,
            embedding_model=embedding_model,
            semantic_threshold=semantic_threshold,
        )
        end_time = time.time()
        logging.info(
            f"Time taken to split the documents into chunks: {end_time-start_time} seconds"
        )
    else:
        total_chunks = clone_database_chunks(
            clone_persist, clone_embedding_model, persist_directory
        )
    # we create the vectordb only once the chunks are cleaned and prepared to avoid creaating incomplete vectordb
    vectordb = load_vectordb(config)

    # Add the documents to the database
    logging.info("Adding documents to the dense vectorstore ...")
    start_time = time.time()
    add_documents_to_db(
        total_chunks,
        vectordb,
        embedding_model,
        persist_directory,
        vectordb_provider=vectordb_provider,
    )
    end_time = time.time()
    logging.info(
        f"Time taken to add the documents to the vectorstore and compute embeddings: {end_time-start_time} seconds"
    )

    del vectordb

    # add the documents to the sparse vectorstore
    logging.info("Adding documents to the sparse vectorstore ...")
    start_time = time.time()
    initialize_sparse_vectorstore(config=config)
    end_time = time.time()
    logging.info(
        f"Time taken to create and add the documents to the sparse vectorstore: {end_time-start_time} seconds"
    )

    if build_knowledge_graph:
        from src.knowledge_graph import KnowledgeGraphIndex
        from src.utils import get_all_docs_qdrant

        logging.info("Building the knowledge graph index ...")
        # initialise the knowledge graph vectorstore object
        start_time = time.time()
        kg_index = KnowledgeGraphIndex(config=config)

        # print("TYpe of total_chunks:", type(total_chunks))
        # print("Length of total_chunks:", len(total_chunks))
        # print("Total chunk first element:", total_chunks[0])

        embedding_model = get_embedding_model(
            model_name=config["embedding_model"]
        )

        # separate clients to avoid conflicts !
        # raw_database = Qdrant.from_existing_collection(
        #     path=config["persist_directory"],
        #     embedding=embedding_model,
        #     collection_name="qdrant_vectorstore",
        # )

        # raw_database=get_existing_qdrant(config["persist_directory"])
        raw_database = get_existing_qdrant(
            persist_directory=persist_directory,
            embedding_model_name=config["embedding_model"],
        )

        # get all the documents from the database instead of reproccessing them
        total_chunks = get_all_docs_qdrant(raw_database=raw_database)

        # fill the knowledge graph with the documents
        kg_index.from_documents(documents=total_chunks, overwrite=True)
        end_time = time.time()
        logging.info(
            f"Time taken to build the knowledge graph index: {end_time-start_time} seconds"
        )

    return total_chunks


@log_execution_time
def get_log_file_path(persist_directory, process_log_file):
    """
    Get the path of the log file.

    This function takes in the persist_directory and process_log_file parameters
    and returns the path of the log file. If the log file does not exist, it creates
    an empty log file.

    Args:
        persist_directory (str): The directory where the log file should be located.
        process_log_file (str): The name of the log file.

    Returns:
        str: The path of the log file.

    """
    log_files = [f for f in os.listdir(persist_directory) if f.endswith(".log")]
    log_file_path = (
        os.path.join(persist_directory, process_log_file)
        if not log_files
        else os.path.join(persist_directory, log_files[0])
    )
    print("Log file path:", log_file_path)
    if not os.path.exists(log_file_path):
        print("Creating log file because it does not exist.")
        with open(log_file_path, "w") as file:
            pass
    return log_file_path


@log_execution_time
def load_processed_docs(log_file_path):
    """
    Load the list of names of the already processed documents from a log file.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        list: The list of already processed documents.

    """
    with open(log_file_path, "r") as file:
        processed_docs = file.read().splitlines()
        # print(f"Already processed docs: {processed_docs}")
        print("Number of already processed documents:", len(processed_docs))
    return processed_docs


@log_execution_time
def load_vectordb(config):
    """
    Load the vectordb from the specified persist directory.

    Args:
        config (dict): The directory where the vectordb is persisted.

    Returns:
        vectordb: The loaded vectordb if it exists, otherwise None.
    """
    # get parameters from config
    vectordb_provider = config["vectordb_provider"]
    persist_directory = config["persist_directory"]

    # get the embedding model
    embeddings = get_embedding_model(model_name=config["embedding_model"])

    if vectordb_provider == "Chroma":
        vectordb = (
            Chroma.load(persist_directory=persist_directory)
            if os.path.exists(os.path.join(persist_directory, "chroma.db"))
            else None
        )

        print("Chroma database loaded successfully!")
        return vectordb

    elif vectordb_provider == "Qdrant":
        if os.path.exists(os.path.join(persist_directory, "collection")):
            print("Collection exists, loading the Qdrant database...")
            qdrant = Qdrant.from_existing_collection(
                path=persist_directory,
                embedding=embeddings,
                collection_name="qdrant_vectorstore",
            )
            return qdrant
        else:
            return None

    return vectordb


def process_file(args):
    """
    Process a file based on the given arguments.

    Args:
        args (tuple): A tuple containing the following elements:
            - name (str): The name of the file.
            - full_path (str): The full path of the file.
            - allowed_formats (list): A list of allowed file formats.
            - processed_docs (list): A list of already processed documents.
            - log_file_path (str): The path to the log file.

    Returns:
        object: The processed document if successful, None otherwise.
    """
    name, full_path, allowed_formats, processed_docs, log_file_path = args
    if (
        name.split(".")[-1] in allowed_formats
        and name not in processed_docs
        and name != "meta.json"
    ):
        loader = find_loader(name, full_path)
        try:
            doc = loader.load()
            with open(log_file_path, "a") as file:
                file.write(name + "\n")
            return doc
        except Exception as e:
            print(f"Error processing {name}: {e}")
    return None


def process_documents(path, log_file_path, processed_docs):
    """
    Process documents by loading and splitting them into chunks, all the files in the directory are processed recursively.

    Args:
        path (str): The path to the directory containing the documents.
        chunk_size (int): The size of each chunk in number of characters.
        chunk_overlap (int): The overlap between consecutive chunks in number of characters.
        log_file_path (str): The path to the log file.
        processed_docs (list): A list to store the processed documents.
        total_chunks (int): The total number of chunks.
        splitting_method (str): The method used for splitting the documents.
        embedding_model (str): The embedding model to use for semantic analysis.
        semantic_threshold (float): The threshold for semantic similarity.

    Returns:
        None
    """

    total_chunks = []

    allowed_formats = json.loads(os.getenv("ALLOWED_FORMATS"))
    # print("ALLOWED FORMATS: ", allowed_formats)

    total_files = sum([len(files) for r, d, files in os.walk(path)])
    with tqdm(total=total_files, desc="Processing files") as pbar:
        with Manager() as manager:
            processed_docs = manager.list(processed_docs)
            pool = Pool()
            args_list = []

            for root, dirs, files in os.walk(path):
                for name in files:
                    full_path = os.path.join(root, name)
                    args_list.append(
                        (
                            name,
                            full_path,
                            allowed_formats,
                            processed_docs,
                            log_file_path,
                        )
                    )

            for result in pool.imap_unordered(process_file, args_list):
                if result:
                    total_chunks.append(result)
                pbar.update()

            pool.close()
            pool.join()

    return total_chunks


@log_execution_time
def filter_and_split_chunks(
    list_documents, embedding_model, semantic_threshold
):
    """
    Filters and prepares chunks based on the specified splitting method, apply pre-processing to the raw documents before chunking, and remove duplicates.

    Args:
        total_chunks (list): A list of Document objects representing the total chunks.
        splitting_method (str): The method used for splitting the chunks. Can be "recursive" or any other value.
        embedding_model (str): The name of the embedding model to be used.
        semantic_threshold (float): The threshold value for semantic chunking.

    Returns:
        list: A list of Document objects representing the filtered and prepared chunks.
    """
    print("STARTING TO SPLIT THE DOCUMENTS INTO CHUNKS ...")

    # load the config file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    chunking_embedding_model = config["chunking_embedding_model"]
    # initialize the semantic splitter with the embedding model
    chunking_embeddings = get_embedding_model(
        model_name=chunking_embedding_model
    )
    semantic_splitter = SemanticChunker(
        embeddings=chunking_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=semantic_threshold,
    )

    # get the total content and metadata of the documents
    total_docs_content = [
        text_preprocessing(chunk[0].page_content) for chunk in list_documents
    ]
    total_docs_metadata = [chunk[0].metadata for chunk in list_documents]
    update_html_sources(total_docs_metadata)
    chunks = semantic_splitter.create_documents(
        texts=total_docs_content, metadatas=total_docs_metadata
    )

    print("SEMANTIC SPLITTING DONE!, STARTING TO REMOVE DUPLICATE CHUNKS ...")
    # remove duplicate chunks
    cleaned_chunks = remove_duplicate_chunks(chunks)

    return cleaned_chunks


def get_modif_date(path):
    # find modif date of the file in its metadata

    temps_modification = os.path.getmtime(path)

    # Convertir le temps en format de date lisible
    date_modification = datetime.fromtimestamp(temps_modification)

    return str(date_modification)


@log_execution_time
def update_html_sources(total_docs_metadata):
    """
    Update the HTML sources in the total_docs_metadata list based on the information
    stored in the html_url_dict.

    Args:
        total_docs_metadata (list): A list of document metadata.

    Returns:
        None
    """
    with open("src/scrapping/url_correspondance.json") as f:
        html_url_dict = json.load(f)
    for i, source in enumerate(total_docs_metadata):
        if ".html" in source and "widgets" not in source:
            raw_file_name = source.split("/")[-1].split(".html")[0]
            try:
                total_docs_metadata[i] = html_url_dict[raw_file_name]
            except:
                pass
        elif ".html" in source and "widgets" in source:
            raw_file_name = (
                source.split("/")[-1].split(".html")[0].split("_widgets")[0]
            )
            try:
                total_docs_metadata[i] = html_url_dict[raw_file_name]
            except:
                pass


def truncate_path_to_data(path):
    """
    Truncates the given path to start from '/data' if '/data' is present in the path.

    Args:
        path (str): The original file path.

    Returns:
        str: The truncated or original path.
    """
    # Check if '/data' is in the path
    if "data/" in path:
        # Find the index of '/data' and truncate everything before it
        data_index = path.index("data/")
        return path[data_index:]
    else:
        # Return the original path if '/data' is not found
        return path


@log_execution_time
def remove_duplicate_chunks(chunks):
    """
    Remove duplicate chunks from a list of chunks and create Document objects, add chunk length and fied to metadata, and clean the content.

    Args:
        chunks (list): A list of Chunk objects.

    Returns:
        list: A list of Document objects created from the unique chunks.

    """
    chunks_content = [chunk.page_content for chunk in chunks]
    seen = set()
    unique_chunks = []
    for chunk, content in zip(chunks, chunks_content):
        if content not in seen:
            seen.add(content)
            unique_chunks.append(chunk)

    total_chunks = []

    for chunk in tqdm(
        unique_chunks, desc="Creating Document objects from the chunks"
    ):
        raw_chunk = chunk.page_content
        chunk_length = len(raw_chunk.split())

        source = truncate_path_to_data(chunk.metadata["source"])

        source_field = source.split("/")[1]  # Get the field from the source

        # if "chatbot_history" in source:
        #     print("Source_field:", source_field)

        # obtain the modification date of the chunk based on the source
        modif_date = get_modif_date(chunk.metadata["source"])

        total_chunks.append(
            Document(
                page_content=raw_chunk,
                metadata={
                    "source": truncate_path_to_data(chunk.metadata["source"]),
                    "chunk_length": chunk_length,
                    "field": source_field,  # Add the new field to the metadata
                    "modif_date": modif_date,
                },
            )
        )
    return total_chunks


@log_execution_time
def clone_database_chunks(
    clone_persist, clone_embedding_model, persist_directory
):
    print("CLONING DATABASE!")
    chunks, metadata = get_k_random_chunks(
        k=1,
        config=None,
        get_all=True,
        clone_persist=clone_persist,
        clone_embedding_model=clone_embedding_model,
    )
    total_chunks = [
        Document(
            page_content=chunk,
            metadata={
                "source": metadata[i]["source"],
                "chunk_length": metadata[i]["chunk_length"],
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    print("CLONING DONE!")
    json_files = [f for f in os.listdir(clone_persist) if f.endswith(".json")]
    if json_files:
        json_file_path = os.path.join(clone_persist, json_files[0])
        base_name = os.path.basename(persist_directory.rstrip("/"))
        new_file_name = str(base_name) + "_evaluation_dataset.json"
        new_file_path = os.path.join(persist_directory, new_file_name)
        shutil.copy(json_file_path, new_file_path)
        print(
            f"Evaluation dataset json file copied to the persist directory as {new_file_name}!"
        )
    return total_chunks


@log_execution_time
def add_documents_to_db(
    total_chunks,
    vectordb,
    embedding_model,
    persist_directory,
    vectordb_provider,
):
    """
    Adds documents to the vector database.

    Args:
        total_chunks (list): A list of documents to be added.
        vectordb (object): The vector database object.
        embedding_model (str): The name of the embedding model.
        persist_directory (str): The directory to persist the vector database.
        vectordb_provider (str): The provider of the vector database.

    Returns:
        None
    """

    # Configure logging for the process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="vector_store_building.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s",
    )

    logging.info("TOTAL NUMBER OF CHUNKS GENERATED:" + str(len(total_chunks)))
    if len(total_chunks) == 0:
        logging.info(
            "No chunks to add to the vectorstore they are already there!"
        )

    logging.info("Starting to add documents to the dense vectorstore ...")
    if total_chunks:
        embedding_model = get_embedding_model(model_name=embedding_model)
        if vectordb is None:
            logging.info("Vectordb is None, creating a new one...")
            if vectordb_provider == "Chroma":
                vectordb = Chroma.from_documents(
                    documents=total_chunks,
                    embedding=embedding_model,
                    persist_directory=persist_directory,
                )
            elif vectordb_provider == "Qdrant":
                vectordb = Qdrant.from_documents(
                    documents=total_chunks,
                    embedding=embedding_model,
                    path=persist_directory,
                    collection_name="qdrant_vectorstore",
                )

        else:
            logging.info(
                "Vectordb is not None, adding documents to the existing database..."
            )
            if vectordb_provider == "Chroma":
                vectordb.add_documents(
                    documents=total_chunks, embedding=embedding_model
                )
            elif vectordb_provider == "Qdrant":
                vectordb.add_documents(documents=total_chunks)

        logging.info("Chroma database fully updated with new documents!")


@log_execution_time
def add_documents_to_db_V2(
    total_chunks,
    vectordb,
    embedding_model,
    sparse_embedding_model,
    persist_directory,
):
    """
    Adds documents to the vector database  (V2 means we create sparse and dense vectorstores in a single vectorstore). Do not support Chroma !

    Args:
        total_chunks (list): A list of documents to be added.
        vectordb (object): The vector database object.
        embedding_model (str): The name of the embedding model.
        persist_directory (str): The directory to persist the vector database.
        vectordb_provider (str): The provider of the vector database.

    Returns:
        None
    """

    # Configure logging for the process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="vector_store_building.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s",
    )

    logging.info("TOTAL NUMBER OF CHUNKS GENERATED:" + str(len(total_chunks)))
    if len(total_chunks) == 0:
        logging.info(
            "No chunks to add to the vectorstore they are already there!"
        )

    logging.info("Starting to add documents to the dense vectorstore ...")
    if total_chunks:
        # Get dense embedding model
        dense_embedding_model = get_embedding_model(model_name=embedding_model)
        # Get sparse embedding model
        sparse_embedding_model = get_sparse_embedding_model(
            model_name=embedding_model
        )

        if vectordb is None:
            logging.info("Vectordb is None, creating a new one...")

            vectordb = QdrantVectorStore.from_documents(
                documents=total_chunks,
                embedding=dense_embedding_model,
                sparse_embedding=sparse_embedding_model,
                path=persist_directory,
                collection_name="qdrant_vectorstore",
                retrieval_mode=RetrievalMode.HYBRID,
            )

        else:
            logging.info(
                "Vectordb is not None, adding documents to the existing database..."
            )

            vectordb.add_documents(documents=total_chunks)

        logging.info("Chroma database fully updated with new documents!")


def find_loader(name, full_path):
    """
    Returns the appropriate document loader based on the file extension.
    input:
        name: the name of the file. type: string
    output:
        loader: the appropriate document loader. type: DocumentLoader
    """
    if name.endswith("pdf"):
        loader = PyPDFLoader(full_path)
    elif name.endswith("docx"):
        loader = Docx2txtLoader(full_path)
    elif name.endswith("xlsx"):
        loader = StructuredExcelLoader(full_path)
    elif name.endswith("html"):
        loader = UnstructuredHTMLLoader(full_path)
    elif name.endswith("pptx"):
        loader = UnstructuredPowerPointLoader(full_path)
    elif name.endswith("txt"):
        loader = TextLoader(full_path)
    elif name.endswith("md"):
        loader = UnstructuredMarkdownLoader(full_path)
    elif name.endswith("json"):
        loader = JSONLoader(full_path, ".", content_key="answer")
    else:
        ext = name.split(".")[-1]
        raise ValueError(
            "Unsupported file format !, concerned file:",
            name,
            "EXTENSION ISSUE:",
            ext,
            "TEST:",
            str(ext in ["pdf", "docx", "html", "pptx", "txt", "md", "json"]),
        )

    return loader


def apply_field_filter_qdrant(search_kwargs, field_filter, field_filter_type):
    """
    Apply a field filter to the search_kwargs dictionary based on the given field_filter and field_filter_type.

    Args:
        search_kwargs (dict): The search keyword arguments dictionary.
        field_filter (list or str): The field filter(s) to apply.
        field_filter_type (str): The type of filter to apply ('$ne' for not equal, '$eq' for equal).

    Returns:
        dict: The updated search_kwargs dictionary with the field filter applied.
    """
    print("FIELD FILTER ENABLED!")

    # Helper to create filter conditions
    def create_filter_condition(field, value, filter_type):
        if filter_type == "$ne":
            return {"must_not": [{"key": field, "match": {"value": value}}]}
        elif filter_type == "$eq":
            return {"must": [{"key": field, "match": {"value": value}}]}
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    # Initialize or update the existing filter
    if "filter" not in search_kwargs:
        search_kwargs["filter"] = {}

    existing_filter = search_kwargs["filter"]

    # Build filters based on input type
    if isinstance(field_filter, list) and len(field_filter) > 1:
        conditions = []
        for field in field_filter:
            condition = create_filter_condition(
                "field", field, field_filter_type
            )
            conditions.append(condition)
        if field_filter_type == "$ne":
            if "should" in existing_filter:
                existing_filter["should"].extend(conditions)
            else:
                existing_filter["should"] = conditions
        elif field_filter_type == "$eq":
            if "must" in existing_filter:
                existing_filter["must"].extend(conditions)
            else:
                existing_filter["must"] = conditions
    else:
        field = (
            field_filter if isinstance(field_filter, str) else field_filter[0]
        )
        condition = create_filter_condition("field", field, field_filter_type)
        if field_filter_type == "$ne":
            if "must_not" in existing_filter:
                existing_filter["must_not"].extend(condition["must_not"])
            else:
                existing_filter["must_not"] = condition["must_not"]
        elif field_filter_type == "$eq":
            if "must" in existing_filter:
                existing_filter["must"].extend(condition["must"])
            else:
                existing_filter["must"] = condition["must"]

    print(
        "Searching for the following fields:",
        field_filter,
        "with filter type:",
        field_filter_type,
    )
    return search_kwargs


def apply_source_filter_qdrant(
    search_kwargs, source_filter, source_filter_type
):
    """
    Apply a source filter to the search_kwargs dictionary based on the given source_filter and source_filter_type.

    Args:
        search_kwargs (dict): The search keyword arguments dictionary.
        source_filter (list or str): The source filter(s) to apply. (name of the sources that we dont want/want to include in the search)
        source_filter_type (str): The type of filter to apply ('$ne' for not equal, '$eq' for equal).

    Returns:
        dict: The updated search_kwargs dictionary with the source filter applied.
    """
    print("SOURCE FILTER ENABLED!")

    # Helper to create filter conditions
    def create_filter_condition(field, value, filter_type):
        if filter_type == "$ne":
            return {"must_not": [{"key": field, "match": {"value": value}}]}
        elif filter_type == "$eq":
            return {"must": [{"key": field, "match": {"value": value}}]}
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    # Initialize or update the existing filter
    if "filter" not in search_kwargs:
        search_kwargs["filter"] = {}

    existing_filter = search_kwargs["filter"]

    # Build filters based on input type
    if isinstance(source_filter, list) and len(source_filter) > 1:
        conditions = []
        for source in source_filter:
            condition = create_filter_condition(
                "source", source, source_filter_type
            )
            conditions.append(condition)
        if source_filter_type == "$ne":
            if "should" in existing_filter:
                existing_filter["should"].extend(conditions)
            else:
                existing_filter["should"] = conditions
        elif source_filter_type == "$eq":
            if "must" in existing_filter:
                existing_filter["must"].extend(conditions)
            else:
                existing_filter["must"] = conditions
    else:
        source = (
            source_filter
            if isinstance(source_filter, str)
            else source_filter[0]
        )
        condition = create_filter_condition(
            "source", source, source_filter_type
        )
        if source_filter_type == "$ne":
            if "must_not" in existing_filter:
                existing_filter["must_not"].extend(condition["must_not"])
            else:
                existing_filter["must_not"] = condition["must_not"]
        elif source_filter_type == "$eq":
            if "must" in existing_filter:
                existing_filter["must"].extend(condition["must"])
            else:
                existing_filter["must"] = condition["must"]

    print(
        "Searching for the following sources:",
        source_filter,
        "with filter type:",
        source_filter_type,
    )
    return search_kwargs


@log_execution_time
def apply_source_filter(search_kwargs, source_filter, source_filter_type):
    """
    Apply a source filter to the search_kwargs dictionary based on the given source_filter and source_filter_type.

    Args:
        search_kwargs (dict): The search keyword arguments dictionary.
        source_filter (list or str): The source filter(s) to apply.
        source_filter_type (str): The type of filter to apply ('$ne' for not equal, '$eq' for equal).

    Returns:
        dict: The updated search_kwargs dictionary with the source filter applied.
    """

    print("SOURCE FILTER ENABLED !")
    # we check if source filter is a list of filters or a single filter
    if len(source_filter) > 1:
        filters = []
        for source in source_filter:
            if source_filter_type == "$ne":
                filters.append({"source": {"$ne": source}})
            elif source_filter_type == "$eq":
                filters.append({"source": {"$eq": source}})

        print(
            "Searching for the following sources:",
            source_filter,
            "with filter type:",
            source_filter_type,
        )

        if source_filter_type == "$ne":
            search_kwargs["filter"] = {"$and": filters}
        elif source_filter_type == "$eq":
            search_kwargs["filter"] = {"$or": filters}

    else:
        if source_filter_type == "$ne":
            search_kwargs["filter"] = {"source": {"$ne": source_filter[0]}}
        elif source_filter_type == "$eq":
            search_kwargs["filter"] = {"source": {"$eq": source_filter[0]}}

        print(
            "Searching for the following sources:",
            source_filter[0],
            "with filter type:",
            source_filter_type,
        )
    return search_kwargs


def get_filtering_kwargs_chromadb(
    source_filter: Union[list, str],
    source_filter_type: str,
    field_filter: Union[list, str],
    field_filter_type: str,
    length_threshold=None,
):
    """
    Get the kwargs that will be used to filter the search based on both the source and the field and chunk length.

    Args:
        source_filter (list or str): The source filter(s) to apply.
        source_filter_type (str): The type of filter to apply ('$ne' for not equal, '$eq' for equal, '$in' for inclusion, etc.).
        field_filter (list or str): The field filter(s) to apply.
        field_filter_type (str): The type of filter to apply ('$ne' for not equal, '$eq' for equal, '$in' for inclusion, etc.).
        length_threshold (float, optional): The minimum length of the chunks to return.

    Returns:
        dict: The updated search_kwargs dictionary with the source and field filters applied.
    """
    search_kwargs = {"where": {}}

    # Helper to create filter conditions
    def create_filter_condition(field, value, filter_type):
        return {field: {filter_type: value}}

    # Build filters based on input type for source
    if isinstance(source_filter, list):
        if source_filter_type == "$in":
            search_kwargs["where"]["metadata.source"] = {
                source_filter_type: source_filter
            }
        else:
            for source in source_filter:
                condition = create_filter_condition(
                    "metadata.source", source, source_filter_type
                )
                search_kwargs["where"].update(condition)
    else:
        condition = create_filter_condition(
            "metadata.source", source_filter, source_filter_type
        )
        search_kwargs["where"].update(condition)

    # Build filters based on input type for field
    if isinstance(field_filter, list):
        if field_filter_type == "$in":
            search_kwargs["where"]["metadata.field"] = {
                field_filter_type: field_filter
            }
        else:
            for field in field_filter:
                condition = create_filter_condition(
                    "metadata.field", field, field_filter_type
                )
                search_kwargs["where"].update(condition)
    else:
        condition = create_filter_condition(
            "metadata.field", field_filter, field_filter_type
        )
        search_kwargs["where"].update(condition)

    # Add length filter if length_threshold is specified
    if length_threshold is not None:
        search_kwargs["where"]["metadata.chunk_length"] = {
            "$gte": float(length_threshold)
        }

    print(
        "Searching for the following sources:",
        source_filter,
        "with filter type:",
        source_filter_type,
    )
    print(
        "Searching for the following fields:",
        field_filter,
        "with filter type:",
        field_filter_type,
    )
    return search_kwargs


def get_filtering_kwargs_qdrant(
    source_filter: list,
    source_filter_type: str,
    field_filter: list,
    field_filter_type: str,
    length_threshold=None,
):
    """
    Generate filtering keyword arguments for querying data in Qdrant.

    Args:
        source_filter (list): A list of source values to filter on.
        source_filter_type (str): The filter type for source values. Can be "$eq" (equal) or "$ne" (not equal).
        field_filter (list): A list of field values to filter on.
        field_filter_type (str): The filter type for field values. Can be "$eq" (equal) or "$ne" (not equal).
        length_threshold (float, optional): The minimum chunk length threshold. Defaults to None.

    Returns:
        dict: A dictionary containing the filtering keyword arguments for querying data in Qdrant.
    """

    search_kwargs = {}

    def create_filter_condition(field, value):
        """
        Create a filter condition based on the given field and value.

        Args:
            field (str): The field to filter on.
            value (Any): The value to match against.

        Returns:
            qdrant_models.FieldCondition: The filter condition object.
        """
        return qdrant_models.FieldCondition(
            key=field, match=qdrant_models.MatchValue(value=value)
        )

    conditions_should = []
    conditions_must = []
    conditions_not = []

    if source_filter:
        for source in source_filter:
            if source_filter_type == "$eq":
                conditions_should.append(
                    create_filter_condition("metadata.source", source)
                )
            else:
                conditions_not.append(
                    create_filter_condition("metadata.source", source)
                )

    if field_filter:
        if len(field_filter) == 1:
            # If there's only one field filter, it becomes a must condition
            field = field_filter[0]

            if field_filter_type == "$eq":
                conditions_should.append(
                    create_filter_condition("metadata.field", field)
                )
            else:
                conditions_not.append(
                    create_filter_condition("metadata.field", field)
                )
        else:
            for field in field_filter:
                if field_filter_type == "$eq":
                    conditions_should.append(
                        create_filter_condition("metadata.field", field)
                    )
                else:
                    conditions_not.append(
                        create_filter_condition("metadata.field", field)
                    )

    if length_threshold is not None:
        # if there is no field filter, we add the length filter as a should condition
        if not field_filter:
            conditions_should.append(
                qdrant_models.FieldCondition(
                    key="metadata.chunk_length",
                    range=qdrant_models.Range(gte=float(length_threshold)),
                )
            )
        else:
            conditions_must.append(
                qdrant_models.FieldCondition(
                    key="metadata.chunk_length",
                    range=qdrant_models.Range(gte=float(length_threshold)),
                )
            )

    search_kwargs["filter"] = qdrant_models.Filter(
        should=conditions_should, must=conditions_must, must_not=conditions_not
    )

    return search_kwargs


@log_execution_time
def apply_multi_query_retriever(base_retriever, nb_chunks):
    """
    Applies a multi-query retriever to the given base retriever.

    Args:
        base_retriever: The base retriever object.
        nb_chunks: The number of chunks to generate queries from the main query.

    Returns:
        vector_db_multi: The multi-query retriever object.

    """
    # we generate several queries from the main query
    chat_model_object = CustomChatModel(
        llm_name="llama3", llm_provider="ollama"
    )
    llm = chat_model_object.chat_model
    vector_db_multi = CustomMultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,
        top_k=nb_chunks,
    )
    return vector_db_multi


@log_execution_time
def apply_auto_hybrid_search(query):
    """
    Applies auto hybrid search to the given query.

    Args:
        query (str): The query to be processed.

    Returns:
        tuple: A tuple containing the hybrid search flag and the word filter.
            - hybrid_search (bool): True if hybrid search is enabled, False otherwise.
            - word_filter (str): The word filter extracted for lexical queries, None for semantic queries.
    """
    print("AUTO HYBRID SEARCH TRIGGERED !")
    query_router = QueryRouter()
    query_router.load()
    label = query_router.predict_label(query)
    hybrid_search = False
    word_filter = None
    if label == "semantic":
        print("QUERY CLASSIFIED AS SEMANTIC !")
    else:
        print("QUERY CLASSIFIED AS LEXICAL, ENABLING HYBRID SEARCH !")
        hybrid_search = True
        word_filter = keywords_extraction(query)
        word_filter = word_filter[0][0]
        print("AUTOMATIC EXTRACTED WORD FILTER:", str(word_filter))
    return hybrid_search, word_filter


@lru_cache(maxsize=None)
def load_client(custom_persist):
    """
    Load the Qdrant client.

    Args:
        custom_persist (str): The custom persist directory.

    Returns:
        QdrantClient: The Qdrant client object.
    """
    client = QdrantClient(path=custom_persist)
    return client


@lru_cache(maxsize=None)
def load_router():
    """
    Load the query router object.

    Returns:
        QueryRouter: The query router object.
    """
    query_router = QueryRouter()
    query_router.load()
    return query_router


@log_execution_time
def apply_advanced_hybrid_search_v2(
    base_retriever,
    nb_chunks,
    query,
    search_kwargs,
):
    """
    Applies advanced hybrid search by filtering and retrieving documents based on various criteria.

    Args:
        base_retriever (Retriever): The base retriever used for retrieval.
        nb_chunks (int): The number of chunks to retrieve.
        length_threshold (int): The minimum length threshold for chunks to be included.
        enable_source_filter (bool): Whether to enable source filtering.
        source_filter_type (str): The type of source filter to apply.
        source_filter (str): The source to filter by.
        query (str): The query to search for.

    Returns:
        Retriever: The hybrid retriever with the filtered and retrieved documents.
    """

    # open the config file
    config = yaml.safe_load(open("config/config.yaml"))
    # get all the chunks

    # obtain the alpha value from the query
    if config["enable_routing"]:
        query_router = load_router()
        label = query_router.predict_label(query)
        alpha = query_router.predict_alpha(query)
        try:
            st.toast("Query type: " + str(label), icon="🔍")
        except:
            pass
    else:
        label = "NO LABEL, USING DEFAULT ALPHA"
        # print("ROUTING DISABLED !")
        alpha = config["alpha"]

    get_vram_logging()
    logging.info("PREDICTED ALPHA: %s", alpha)
    logging.info("PREDICTED LABEL: %s", label)

    custom_persist = config["persist_directory"] + "_sparse_vector"

    start_time = time.time()
    client = load_client(custom_persist)
    end_time = time.time()
    get_vram_logging()
    logging.info("TIME TO LOAD QDRANT CLIENT: %s", end_time - start_time)

    import functools

    # use functools to make a callable computed_sparse_vector function
    compute_sparse_vector_callable = functools.partial(
        compute_sparse_vector, model_id=config["sparse_embedding_model"]
    )

    start_time = time.time()
    sparse_retriever = QdrantSparseVectorRetriever(
        k=nb_chunks,  # the number of chunks to return
        filter=search_kwargs[
            "filter"
        ],  # the object here should be a qdrant_models.Filter object !!
        client=client,
        collection_name="sparse_vector",
        sparse_vector_name="text",
        sparse_encoder=compute_sparse_vector_callable,
        # compute_sparse_vector(model_id=config['sparse_embedding_model']),
    )

    end_time = time.time()
    get_vram_logging()
    logging.info("TIME TO LOAD SPARSE RETRIEVER: %s", end_time - start_time)

    # Initialize the ensemble retriever
    base_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, base_retriever],
        weights=[alpha, 1 - alpha],
    )

    return base_retriever


@log_execution_time
def apply_advanced_hybrid_search_v3(
    base_retriever,
    nb_chunks,
    query,
    search_kwargs,
):
    """
    Applies advanced hybrid search by filtering and retrieving documents based on various criteria (V3 is using a unique vectorstore instead of two)

    Args:
        base_retriever (Retriever): The base retriever used for retrieval.
        nb_chunks (int): The number of chunks to retrieve.
        length_threshold (int): The minimum length threshold for chunks to be included.
        enable_source_filter (bool): Whether to enable source filtering.
        source_filter_type (str): The type of source filter to apply.
        source_filter (str): The source to filter by.
        query (str): The query to search for.

    Returns:
        Retriever: The hybrid retriever with the filtered and retrieved documents.
    """

    # open the config file
    config = yaml.safe_load(open("config/config.yaml"))
    # get all the chunks

    # obtain the alpha value from the query
    if config["enable_routing"]:
        query_router = load_router()
        label = query_router.predict_label(query)
        alpha = query_router.predict_alpha(query)
        try:
            st.toast("Query type: " + str(label), icon="🔍")
        except:
            pass
    else:
        label = "NO LABEL, USING DEFAULT ALPHA"
        # print("ROUTING DISABLED !")
        alpha = config["alpha"]

    from qdrant_client.models import FusionQuery, RrfParamsMap

    hybrid_fusion = FusionQuery(
        fusion=models.Fusion.RRF,
        params=RrfParamsMap(
            dense=alpha,  # Poids pour la recherche dense
            sparse=1 - alpha,  # Poids pour la recherche sparse
        ),
    )

    get_vram_logging()
    logging.info("PREDICTED ALPHA: %s", alpha)
    logging.info("PREDICTED LABEL: %s", label)

    # load the QdrantVectorStore from existing vectorstore
    client = load_client(config["persist_directory"])

    start_time = time.time()
    hybrid_retriever = QdrantVectorRetriever(
        k=nb_chunks,  # the number of chunks to return
        filter=search_kwargs[
            "filter"
        ],  # the object here should be a qdrant_models.Filter object !!
        client=client,
        collection_name="hybrid_vector",
        # sparse_vector_name="text",
        # sparse_encoder=compute_sparse_vector_callable,
        # compute_sparse_vector(model_id=config['sparse_embedding_model']),
    )

    end_time = time.time()
    get_vram_logging()
    logging.info("TIME TO LOAD SPARSE RETRIEVER: %s", end_time - start_time)
    return base_retriever


@lru_cache(maxsize=None)
def load_reranker(model_name, device="cuda", show_progress=False):
    """
    Load the reranker model based on the given model name and device.

    Args:
        model_name (str): The name of the model.
        device (str): The device to load the model on. Default is "cuda".
        show_progress (bool): Whether to show progress bars during model loading. Default is False.

    Returns:
        Reranker: The reranker model object.
    """
    # Disable transformers logging and progress bars if show_progress is False
    if not show_progress:
        transformers_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)

    model_kwargs = (
        {
            "automodel_args": {"torch_dtype": torch.float16},
            "device": device,
            "trust_remote_code": True,
            "max_length": 1024,  # crucial for the jina reranker !!!
        }
        if "jina" in model_name.lower()
        else {"device": device}
    )

    reranker = HuggingFaceCrossEncoder(
        model_name=model_name, model_kwargs=model_kwargs
    )

    return reranker


@log_execution_time
def apply_reranker(
    query=None,
    base_retriever=None,
    config=None,
):
    """
    Applies a reranker to enhance precision and compresses documents before reranking.

    Args:
        base_retriever: The base retriever to use.
        query (str): The query to retrieve relevant documents.
        config (dict): The configuration dictionary.

    Returns:
        compressed_docs: The compressed documents.

    """
    # Get the parameters from the config file
    reranker_model = config["reranker_model"]
    nb_rerank = config["nb_rerank"]
    nb_chunks = config["nb_chunks"]
    token_compression = config["token_compression"]
    llm_token_target = config["llm_token_target"]
    reranker_token_target = config["reranker_token_target"]
    use_autocut = config["use_autocut"]
    autocut_beta = config["autocut_beta"]

    # Load the reranker model
    reranker = load_reranker(reranker_model)
    get_vram_logging()
    logging.info("RERANKER LOADED !")

    # Instantiate CustomCrossEncoderReranker
    if (
        llm_token_target == 0
    ):  # if its None, the reranker outputs nb_rerank documents else it outputs documents until the llm_token_target token objective is reached
        # print("NONE TOKEN TARGET FOR RERANKER !")
        intelligent_compression = False
        # put a default value for the llm_token_target

    else:
        intelligent_compression = True

    reranker_compressor = CustomCrossEncoderReranker(
        model=reranker,
        top_n=nb_rerank,
        use_autocut=use_autocut,
        autocut_beta=autocut_beta,
        intelligent_compression=intelligent_compression,
        token_target=llm_token_target,
    )

    if (
        reranker_token_target == 0 or reranker_token_target is None
    ):  # if its None, we give to the reranker nb_chunks documents to rerank else we give documents until the reranker_token_target token objective is reached
        intelligent_compression = False
    else:
        intelligent_compression = True

    # Instantiate TopKCompressor and compress documents before reranking

    top_k_compressor = TopKCompressor(
        k=nb_chunks,
        intelligent_compression=intelligent_compression,
        token_target=reranker_token_target,
    )  # if llm_token_target is None, we output the nb_chunks documents without compression

    if token_compression:  # if we use the token_compressor with chain the compressors (token_compressor before reranker !)
        # instantiate reranker compressor

        # Instantiate LLMLinguaCompressor for token compression
        token_compressor = LLMLinguaCompressor(
            model_name="microsoft/phi-2", device_map="cpu", target_token=2000
        )  # ATTENTION: we use CPU !

        pipeline_compressor = DocumentCompressorPipeline(  # THIS IS CHAINING THE COMPRESSORS TOP K THEN RERANKER
            transformers=[
                top_k_compressor,
                reranker_compressor,
                token_compressor,
            ]
        )

    else:  # if we don't use the token_compressor
        pipeline_compressor = DocumentCompressorPipeline(  # THIS IS CHAINING THE COMPRESSORS TOP K THEN RERANKER
            transformers=[top_k_compressor, reranker_compressor]
        )

    # We connect the compressors pipeline to the base retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever,
    )

    # if token_compression:  # if we use the token_compressor with chain the compressors (token_compressor before reranker !)
    #     compression_retriever = ContextualCompressionRetriever(
    #         base_compressor=pipeline_compressor,
    #         base_retriever=token_compression_retriever,
    #     )
    # else:  # if we don't use the token_compressor
    #     compression_retriever = ContextualCompressionRetriever(
    #         base_compressor=pipeline_compressor,
    #         base_retriever=base_retriever,
    #     )

    # print("APPLYING WHOLE PIPELINE ...")
    # We get the top_n most relevant documents
    compressed_docs = compression_retriever.get_relevant_documents(query=query)

    return compressed_docs


@log_execution_time
def apply_auto_merging(
    auto_merging,
    auto_merging_threshold,
    compressed_docs,
    raw_database,
    search_type,
    query,
    text_preprocessing,
):
    """
    Apply auto merging to retrieve a full document if a certain threshold of subchunks are from the same source.

    Args:
        auto_merging (bool): Flag indicating whether auto merging should be applied.
        auto_merging_threshold (int): The threshold for the number of chunks from the same source required for auto merging.
        compressed_docs (list): List of compressed documents.
        raw_database: The raw database object used for retrieval.
        search_type: The type of search to be performed.
        query: The query used for retrieval.
        text_preprocessing: The text preprocessing function to be applied to each chunk.

    Returns:
        list: The updated list of compressed documents after applying auto merging.
    """
    if auto_merging:
        source_doc_counts = defaultdict(int)
        for doc in compressed_docs:
            source_doc_counts[doc.metadata["source"]] += 1

        for source_doc, count in source_doc_counts.items():
            if count >= auto_merging_threshold:
                print("AUTO MERGING TRIGGERED !")
                search_kwargs = {"k": 100, "filter": {"source": source_doc}}
                source_doc_finder = raw_database.as_retriever(
                    search_kwargs=search_kwargs, search_type=search_type
                )
                same_source_chunks = source_doc_finder.get_relevant_documents(
                    query=query
                )
                for doc in same_source_chunks:
                    doc.page_content = text_preprocessing(doc.page_content)
                content_merged = " ".join(
                    [doc.page_content for doc in same_source_chunks]
                )
                chunk_lengths = [
                    int(doc.metadata["chunk_length"])
                    for doc in same_source_chunks
                ]
                merged_length = sum(chunk_lengths)
                merged_doc = Document(
                    page_content=content_merged,
                    metadata={
                        "source": source_doc,
                        "chunk_length": merged_length,
                    },
                )
                compressed_docs = [
                    doc
                    for doc in compressed_docs
                    if doc.metadata["source"] != source_doc
                ]
                compressed_docs.append(merged_doc)

    return compressed_docs


@lru_cache(maxsize=None)
def get_existing_qdrant(persist_directory, embedding_model_name):
    """
    Get an existing Qdrant database from the specified directory.

    Args:
        persist_directory (str): The directory where the Qdrant database is stored.
        embedding_model_name (str): The name of the embedding model used in the database.

    Returns:
        Qdrant: The existing Qdrant database.

    """

    # load the embedding model
    embedding_model = get_embedding_model(model_name=embedding_model_name)

    database = Qdrant.from_existing_collection(
        path=persist_directory,
        embedding=embedding_model,
        collection_name="qdrant_vectorstore",
    )
    return database


@traceable
@log_execution_time
def query_database_v2(query, default_config, config={}):
    """
    Query the database using the specified configuration.

    Args:
        query (str): The query string.
        default_config (dict): The default configuration.
        config (dict, optional): The additional configuration. Defaults to {}.

    Returns:
        list: The list of compressed documents. format: [Document, Document, ...]

    """

    # logging.basicConfig(filename='query_database.log',filemode='a', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # Merge default_config and config. Values in config will override those in default_config.
    config = {**default_config, **config}

    # we load the embedding model
    embedding_model = get_embedding_model(model_name=config["embedding_model"])

    # We define the raw database
    if "chroma" in config["persist_directory"]:
        raw_database = Chroma(
            persist_directory=config["persist_directory"],
            embedding_function=embedding_model,
        )  # database not in retriever form
    elif "qdrant" in config["persist_directory"]:
        get_vram_logging()
        logging.info("Trying to load the qdrant database...")
        raw_database = get_existing_qdrant(
            persist_directory=config["persist_directory"],
            embedding_model_name=config["embedding_model"],
        )

    # we define the search kwargs
    if config["vectordb_provider"] == "Chroma":
        search_kwargs = {"k": config["nb_chunks"]}
        if config["filter_on_length"]:  # we filter on the length of the chunks
            search_kwargs["filter"] = {
                "chunk_length": {"$gt": config["length_threshold"]}
            }

        if config["enable_source_filter"]:  # we filter on the metadata
            search_kwargs = apply_source_filter(
                search_kwargs,
                config["source_filter"],
                config["source_filter_type"],
            )
            logging.info("FINAL_SEARCH_KWARGS:", search_kwargs)

        # if auto_hybrid_search is True, we extract the first word of the query and use it as a filter
        if config["auto_hybrid_search"]:
            config["hybrid_search"], config["word_filter"] = (
                apply_auto_hybrid_search(query)
            )

        # if hybrid_search is True, we use the hybrid search
        if config["hybrid_search"]:  # we use the hybrid search
            search_kwargs["where_document"] = {
                "$contains": config["word_filter"]
            }

    elif config["vectordb_provider"] == "Qdrant":
        get_vram_logging()
        logging.info("USING QDRANT SEARH KWARGS !")
        search_kwargs = get_filtering_kwargs_qdrant(
            source_filter=config["source_filter"],
            source_filter_type=config["source_filter_type"],
            field_filter=config["field_filter"],
            field_filter_type=config["field_filter_type"],
            length_threshold=config["length_threshold"],
        )

        # add a 'k' key to the search_kwargs dictionary
        search_kwargs["k"] = config["nb_chunks"]

        logging.info("QUERY: %s", query)
        logging.info("USED SEARCH KWARGS: %s", search_kwargs)

        # if auto_hybrid_search is True, we extract the first word of the query and use it as a filter
        if config["auto_hybrid_search"]:
            config["hybrid_search"], config["word_filter"] = (
                apply_auto_hybrid_search(query)
            )

        # if hybrid_search is True, we use the hybrid search
        if config["hybrid_search"]:  # we use the hybrid search
            search_kwargs["where_document"] = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="word_filter",
                        match=qdrant_models.MatchValue(
                            value=config["word_filter"]
                        ),
                    )
                ]
            )

    # DEFINE THE BASE RETRIEVER (Qdrant or Chroma)
    base_retriever = raw_database.as_retriever(
        search_kwargs=search_kwargs, search_type=config["search_type"]
    )

    # BASE RETRIEVER TRANSFORMED INTO MULTI QUERY RETRIEVER
    if config["use_multi_query"]:
        base_retriever = apply_multi_query_retriever(
            base_retriever, config["nb_chunks"]
        )

    # BASE RETRIEVER TRANSFORMED INTO TOKEN COMPRESSION RETRIEVER
    # if config["token_compression"]:
    #     compressor = LLMLinguaCompressor(
    #         model_name="microsoft/phi-2", device_map="cpu", target_token=4000 #ATTENTION: we use the CPU here
    #     )
    #     token_compression_retriever = ContextualCompressionRetriever(
    #         base_compressor=compressor, base_retriever=base_retriever
    #     )

    # LAUNCHING THE ADVANCED HYBRID SEARCH
    if config["advanced_hybrid_search"]:
        base_retriever = apply_advanced_hybrid_search_v2(
            base_retriever=base_retriever,
            nb_chunks=config["nb_chunks"],
            query=query,
            search_kwargs=search_kwargs,
        )

    # APPLYING THE RERANKER
    if config["use_reranker"]:
        compressed_docs = apply_reranker(
            base_retriever=base_retriever,
            query=query,
            config=config,
        )

    else:
        # We get the top_n most relevant documents
        retriever = base_retriever
        compressed_docs = retriever.get_relevant_documents(query=query)

    if config[
        "auto_merging"
    ]:  # we use auto merging to retrive a full document if a certain threshold of subchunks are from the same source
        compressed_docs = apply_auto_merging(
            config["auto_merging"],
            config["auto_merging_threshold"],
            compressed_docs,
            raw_database,
            config["search_type"],
            query,
            config["text_preprocessing"],
        )
    get_vram_logging()
    logging.info(
        "FIELDS FOUND: %s", [k.metadata["field"] for k in compressed_docs]
    )
    return compressed_docs


@log_execution_time
def query_knowledge_graph_v2(
    query,
    default_config,
    config={},
):
    """
    Query the knowledge graph to retrieve relevant information based on the given query.

    Args:
        query (str): The query string.
        default_config (dict): The default configuration dictionary.
        config (dict, optional): The configuration dictionary to override default_config. Defaults to {}.
        kg_target_context (int, optional): The target number of tokens for the retrieved information. Defaults to 1000.
        min_relations_pct (float, optional): The minimum percentage of tokens for relations. Defaults to 0.1.
        min_descriptions_pct (float, optional): The minimum percentage of tokens for descriptions. Defaults to 0.2.
        min_communities_pct (float, optional): The minimum percentage of tokens for communities. Defaults to 0.2.

    Returns:
        dict: A dictionary containing the selected relations, descriptions, and communities.
    """

    from src.knowledge_graph import load_entity_extractor
    from src.utils import token_calculation_prompt

    # Merge default_config and config. Values in config will override those in default_config.
    config = {**default_config, **config}

    # Get the parameters from the config file
    kg_target_context = config["kg_target_context"]
    min_relations_pct = config["min_relations_pct"]
    min_descriptions_pct = config["min_descriptions_pct"]
    min_communities_pct = config["min_communities_pct"]

    num_nodes = config["nb_nodes"]

    # Load the embedding model
    embedding_model = get_embedding_model(model_name=config["embedding_model"])

    # Initialize the entity extractor to extract entities from the query and perform a similarity search on each entity
    entity_extractor = load_entity_extractor(config["config_path"])

    # Extract entities from the query
    query_entities = entity_extractor.extract_entities(query)

    # list of entities
    list_cleaned_entities = [entity["text"] for entity in query_entities]

    # print a st.toast message for each entity indicating that the entity is searched in the knowledge graph
    if list_cleaned_entities == []:
        st.toast("No entities found in the query...", icon="🔍")
    for entity in list_cleaned_entities:
        try:
            st.toast(
                "Searching for " + entity + " in the knowledge graph...",
                icon="🔍",
            )
        except:
            pass

    # Define the raw database
    if "chroma" in config["persist_directory"]:
        raw_database = Chroma(
            persist_directory=config["persist_directory"],
            embedding_function=embedding_model,
        )  # database not in retriever form
    elif "qdrant" in config["persist_directory"]:
        logging.info("Trying to load the qdrant database...")
        # raw_database = get_existing_qdrant(config["persist_directory"] + "_kg")
        raw_database = get_existing_qdrant(
            persist_directory=config["persist_directory"] + "_kg",
            embedding_model_name=config["embedding_model"],
        )

    def process_entity(entity):
        search_kwargs = {
            "filter": qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.type",
                        match=qdrant_models.MatchValue(value="entity"),
                    )
                ]
            )
        }

        node = raw_database.similarity_search(
            query=entity, k=1, **search_kwargs
        )[0]  # we take the first result of the list

        # Extract the list of relations, the description, and the community of the entity
        found_relations = [
            f"{node.page_content} {relation['relation'].replace('workswith', 'works with').replace('workson', 'works on').replace('contributesto', 'contributes to').replace('makesuseof', 'makes use of').replace('locatedin', 'located in').replace('includedin', 'included in').replace('workfor', 'works for')} {relation['linked_entity']}."
            for relation in node.metadata["relation_list"]
        ]

        found_description = node.metadata["description"]
        found_community_summary = node.metadata["community"]["summary"]

        return found_relations, found_description, found_community_summary

    if list_cleaned_entities != []:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_entity, list_cleaned_entities))

        list_relations, list_description, list_communities = zip(*results)
        list_relations = [
            item for sublist in list_relations for item in sublist
        ]
        list_description = list(list_description)
        list_communities = list(list_communities)

        # Total list of info
        total_info = list_communities + list_relations + list_description

        # Use the reranker to select the most relevant infos to send to the LLM
        reranker = load_reranker(config["reranker_model"])

        # Make a tuple list of pairs query info
        list_description_tuples = [(query, info) for info in total_info]

        # Get the scores for all the infos
        scores = reranker.score(list_description_tuples)

        # Sort the information based on scores
        sorted_info = [
            info for _, info in sorted(zip(scores, total_info), reverse=True)
        ]

        # Initialize the final lists and token count
        selected_relations = []
        selected_descriptions = []
        selected_communities = []
        total_tokens = 0

        # Calculate minimum tokens required for each category
        min_relations_tokens = kg_target_context * min_relations_pct
        min_descriptions_tokens = kg_target_context * min_descriptions_pct
        min_communities_tokens = kg_target_context * min_communities_pct

        # Add documents to the final lists until the target number of tokens is reached
        for info in sorted_info:
            tokens = token_calculation_prompt(info)
            if total_tokens + tokens > kg_target_context:
                break
            if info in list_relations:
                selected_relations.append(info)
            elif info in list_description:
                selected_descriptions.append(info)
            elif info in list_communities:
                selected_communities.append(info)
            total_tokens += tokens

        # Ensure minimum token requirements are met
        def ensure_min_tokens(selected_list, source_list, min_tokens):
            current_tokens = sum(
                token_calculation_prompt(info) for info in selected_list
            )
            for info in source_list:
                if current_tokens >= min_tokens:
                    break
                if info not in selected_list:
                    selected_list.append(info)
                    current_tokens += token_calculation_prompt(info)

        ensure_min_tokens(
            selected_relations, list_relations, min_relations_tokens
        )
        ensure_min_tokens(
            selected_descriptions, list_description, min_descriptions_tokens
        )
        ensure_min_tokens(
            selected_communities, list_communities, min_communities_tokens
        )

        # print the total number of tokens from extracted infos
        print(
            "TOTAL TOKENS KG:",
            sum(
                token_calculation_prompt(info)
                for info in selected_relations
                + selected_descriptions
                + selected_communities
            ),
        )

        return {
            "selected_relations": selected_relations,
            "selected_descriptions": selected_descriptions,
            "selected_communities": selected_communities,
        }
    else:
        return {
            "selected_relations": ["No relations found..."],
            "selected_descriptions": ["No descriptions found..."],
            "selected_communities": ["No communities found..."],
        }


import logging

from src.utils import log_execution_time


@log_execution_time
def query_knowledge_graph_v3(query, default_config, config={}):
    """
    Query the knowledge graph to retrieve relevant information based on the given query.

    This function extracts entities from the query, searches for their relations, descriptions,
    and community summaries, and performs a similarity search using the entire query. The results
    are then reranked using a CustomCrossEncoderReranker with autocut to filter the most relevant
    information.

    Args:
        query (str): The query string.
        default_config (dict): The default configuration dictionary.
        config (dict, optional): The configuration dictionary to override default_config. Defaults to {}.

    Returns:
        dict: A dictionary containing the selected relations, descriptions, and communities.
    """
    config = {**default_config, **config}

    from src.knowledge_graph import load_entity_extractor
    from src.utils import token_calculation_prompt

    # Configure logging for the process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="knowledgraphe_retrieval.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(funcName)s:%(message)s",
    )

    # Load models
    embedding_model = get_embedding_model(model_name=config["embedding_model"])
    entity_extractor = load_entity_extractor(config["config_path"])

    # Extract entities from the query
    query_entities = entity_extractor.extract_entities(query)
    list_cleaned_entities = [entity["text"] for entity in query_entities]

    if list_cleaned_entities == []:
        st.toast("No entities found in the query...", icon="🔍")
    for entity in list_cleaned_entities:
        try:
            st.toast(
                "Searching for " + entity + " in the knowledge graph...",
                icon="🔍",
            )
        except:
            pass

    # Define the raw database
    raw_database = get_existing_qdrant(
        persist_directory=config["persist_directory"] + "_kg",
        embedding_model_name=config["embedding_model"],
    )

    def process_entity(entity):
        """
        Process an entity and retrieve related information.

        Args:
            entity (str): The entity to be processed.

        Returns:
            tuple: A tuple containing the following information:
                - found_relations (list): A list of strings representing the found relations.
                - found_description (str): The description of the entity.
                - found_community_summary (str): The summary of the community related to the entity.
        """
        search_kwargs = {
            "filter": qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.type",
                        match=qdrant_models.MatchValue(value="entity"),
                    )
                ]
            )
        }
        node = raw_database.similarity_search(
            query=entity, k=1, **search_kwargs
        )[0]
        logging.info(f"ENTITY NOTE RETRIEVED: {node}")
        found_relations = [
            f"{node.page_content} {relation['relation'].replace('workswith', 'works with').replace('workson', 'works on').replace('contributesto', 'contributes to').replace('makesuseof', 'makes use of').replace('locatedin', 'located in').replace('includedin', 'included in').replace('workfor', 'works for')} {relation['linked_entity']}."
            for relation in node.metadata["relation_list"]
        ]
        found_description = node.metadata["description"]
        found_community_summary = node.metadata["community"]["summary"]
        return found_relations, found_description, found_community_summary

    selected_relations = []
    selected_descriptions = []
    selected_communities = []

    # We process the entities in parallel
    if list_cleaned_entities != []:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_entity, list_cleaned_entities))

        list_relations, list_description, list_communities = zip(*results)
        list_relations = [
            item for sublist in list_relations for item in sublist
        ]
        list_description = list(list_description)
        list_communities = list(list_communities)

        # Transform the list of relations into a list of langchain Document objects
        list_relations = [
            Document(page_content=relation, metadata={"type": "relation"})
            for relation in list_relations
        ]

        reranker = CustomCrossEncoderReranker(
            model=load_reranker(config["reranker_model"]), use_autocut=True
        )
        selected_relations = reranker.compress_documents(list_relations, query)

    search_kwargs = {
        "filter": qdrant_models.Filter(
            should=[
                qdrant_models.FieldCondition(
                    key="metadata.type",
                    match=qdrant_models.MatchValue(value="community_summary"),
                ),
                qdrant_models.FieldCondition(
                    key="metadata.type",
                    match=qdrant_models.MatchValue(value="entity_description"),
                ),
            ]
        )
    }

    # We now search for the community summaries and entity descriptions based on semantic search using the entire query
    similarity_results = raw_database.similarity_search(
        query=query, k=30, **search_kwargs
    )
    similarity_docs = [result for result in similarity_results]
    reranker = CustomCrossEncoderReranker(
        model=load_reranker(config["reranker_model"]), use_autocut=True
    )
    filtered_docs = reranker.compress_documents(similarity_docs, query)

    selected_descriptions = [
        doc.page_content
        for doc in filtered_docs
        if doc.metadata["type"] == "entity_description"
    ]
    selected_communities = [
        doc.page_content
        for doc in filtered_docs
        if doc.metadata["type"] == "community_summary"
    ]
    # Transform the selected relations into a list of strings
    selected_relations = [doc.page_content for doc in selected_relations]

    # Logging the required information
    logging.info(f"Number of entities reranked: {len(list_cleaned_entities)}")
    logging.info(f"Number of summaries reranked: {len(selected_communities)}")
    logging.info(
        f"Number of descriptions reranked: {len(selected_descriptions)}"
    )

    return {
        "selected_relations": selected_relations,
        "selected_descriptions": selected_descriptions,
        "selected_communities": selected_communities,
    }


if __name__ == "__main__":
    # open the config file
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)
