import json
import logging
import os
import warnings
from datetime import datetime

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import (
    Qdrant,
    QdrantVectorStore,
    RetrievalMode,
)
from tqdm import tqdm
# custom imports 
from src.embedding_model import get_embedding_model, get_sparse_embedding_model
# from src.query_routing_utils import QueryRouter
from src.utils import (
    log_execution_time,
    text_preprocessing,
)

# Load the environment variables (API keys, etc...)
load_dotenv()

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


class VectorAgent:
    def __init__(self, default_config, config={}):
        self.config = {**default_config, **config}
        self.client = None
        self.persist_directory = self.config["persist_directory"]
        self.process_log_file = self.config["process_log_file"]
        self.clone_database = self.config["clone_database"]
        self.dense_embedding_model_name = self.config["embedding_model"]
        self.semantic_threshold = self.config["semantic_threshold"]
        self.clone_persist = self.config["clone_persist"]
        self.clone_embedding_model = self.config["clone_embedding_model"]
        self.path = self.config["path"]
        print("Path:", self.path)
        self.vectordb_provider = self.config["vectordb_provider"]
        self.build_knowledge_graph = self.config["build_knowledge_graph"]
        self.log_file_path = ""
        self.already_processed_docs = []
        self.allowed_formats = json.loads(os.getenv("ALLOWED_FORMATS"))
        self.dense_embedding_model = get_embedding_model(
            model_name=self.dense_embedding_model_name
        )
        self.chunking_embedding_model = get_embedding_model(
            model_name=self.config["chunking_embedding_model"]
        )
        self.sparse_embedding_model = get_sparse_embedding_model(
            model_name=self.config["sparse_embedding_model"]
        )

        self.log_file_name = self.config["process_log_file"]

        # storage variables
        self.total_chunks = []
        self.total_documents = []
        
    def create_persist_directory(self):
        """
        Creates the persist directory if it does not already exist.

        This method checks if the directory specified by `self.persist_directory` exists.
        If it does not exist, it logs a message indicating that the directory is being created,
        creates the directory, and then logs a message indicating that the directory was created successfully.
        """
        if not os.path.exists(self.persist_directory):
            logging.info("Persist directory does not exist!, creating it ...")
            os.makedirs(self.persist_directory)
            logging.info("Persist directory created successfully!")
        


    def get_log_file_path(self):
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
        log_files = [
            f for f in os.listdir(self.persist_directory) if f.endswith(".log")
        ]
        log_file_path = (
            os.path.join(self.persist_directory, self.log_file_name)
            if not log_files
            else os.path.join(self.persist_directory, log_files[0])
        )
        print("Log file path:", log_file_path)
        if not os.path.exists(log_file_path):
            print("Creating log file because it does not exist.")
            with open(log_file_path, "w") as file:
                pass

        self.log_file_path = log_file_path

    def save_config_file(self):
        """
        Save the configuration file used to create the vectorstore.

        This method writes the configuration data to a YAML file named
        "building_config.yaml" in the persist directory.

        Raises:
            OSError: If there is an issue writing the file.
        """
        # Save the config file that was used to create the vectorstore in the persist directory
        with open(
            os.path.join(self.persist_directory, "building_config.yaml"), "w"
        ) as file:
            yaml.dump(config, file)

    def find_already_processed(self):
        """
        Load the list of names of the already processed documents from a log file.

        Args:
            log_file_path (str): The path to the log file.

        Returns:
            list: The list of already processed documents.

        """
        if self.config["use_server"]:
            #no documents are processed yet
            self.already_processed_docs = []
        else:
            with open(self.log_file_path, "r") as file:
                self.already_processed_docs = file.read().splitlines()
                print(
                    "Number of already processed documents:",
                    len(self.already_processed_docs),
                )

    def process_documents(self):
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
            dense_embedding_model (str): The embedding model to use for semantic analysis.
            semantic_threshold (float): The threshold for semantic similarity.

        Returns:
            None
        """

        # get the total number of files in the directory
        total_files = sum([len(files) for r, d, files in os.walk(self.path)])
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for root, dirs, files in os.walk(self.path):
                for name in files:
                    full_path = os.path.join(root, name)
                    result = self.process_single_file(
                        name,
                        full_path,
                    )
                    if result:
                        self.total_documents.append(result) # add the processed document to the total documents
                    pbar.update()


    def process_single_file(self, name, full_path):
        """
        Process a single file based on the given arguments.

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

        if (
            name.split(".")[-1] in self.allowed_formats
            and name not in self.already_processed_docs
            and name != "meta.json"
        ):
            loader = self.find_loader(name, full_path)
            try:
                doc = loader.load()
                try:
                    with open(self.log_file_path, "a") as file:
                        file.write(name + "\n")
                except Exception as e:
                    pass
                return doc
            except Exception as e:
                print(f"Error processing {name}: {e}")
                return None

    def find_loader(self, name, full_path):
        """
        Returns the appropriate document loader based on the file extension.
        input:
            name: the name of the file. type: string
        output:
            loader: the appropriate document loader. type: DocumentLoader
        """
        if name.endswith("pdf"):
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(full_path)
        elif name.endswith("docx"):
            from langchain_community.document_loaders import Docx2txtLoader

            loader = Docx2txtLoader(full_path)
        elif name.endswith("xlsx"):
            from langchain_community.document_loaders import (
                StructuredExcelLoader,
            )

            loader = StructuredExcelLoader(full_path)
        elif name.endswith("html"):
            from langchain_community.document_loaders import (
                UnstructuredHTMLLoader,
            )

            loader = UnstructuredHTMLLoader(full_path)
        elif name.endswith("pptx"):
            from langchain_community.document_loaders import (
                UnstructuredPowerPointLoader,
            )

            loader = UnstructuredPowerPointLoader(full_path)
        elif name.endswith("txt"):
            from langchain_community.document_loaders import TextLoader

            loader = TextLoader(full_path)
        elif name.endswith("md"):
            from langchain_community.document_loaders import (
                UnstructuredMarkdownLoader,
            )

            loader = UnstructuredMarkdownLoader(full_path)
        elif name.endswith("json"):
            from langchain_community.document_loaders import JSONLoader

            loader = JSONLoader(full_path, ".", content_key="answer")
        else:
            ext = name.split(".")[-1]
            raise ValueError(
                "Unsupported file format !, concerned file:",
                name,
                "EXTENSION ISSUE:",
                ext,
                "TEST:",
                str(
                    ext in ["pdf", "docx", "html", "pptx", "txt", "md", "json"]
                ),
            )

        return loader

    @log_execution_time
    def filter_and_split_chunks(self):
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
    
        if self.config["splitting_method"] == "semantic":
            semantic_splitter = SemanticChunker(
                embeddings=self.chunking_embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=self.config["semantic_threshold"],
            )
    
            # get the total content of the documents
            total_docs_content = [
                text_preprocessing(chunk[0].page_content)
                for chunk in self.total_documents
            ]
            # get the total metadata of the documents
            total_docs_metadata = [
                chunk[0].metadata for chunk in self.total_documents
            ]
            
            chunks = semantic_splitter.create_documents(
                texts=total_docs_content, metadatas=total_docs_metadata
            )
        else:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
    
            chunk_size = self.config["chunk_size"] # default chunk size
            chunk_overlap = self.config["chunk_overlap"]  # default chunk overlap
    
            character_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    
            # get the total content of the documents
            total_docs_content = [
                text_preprocessing(chunk[0].page_content)
                for chunk in self.total_documents
            ]
            # get the total metadata of the documents
            total_docs_metadata = [
                chunk[0].metadata for chunk in self.total_documents
            ]
    
            chunks = [
                Document(
                    page_content=content,
                    metadata=metadata
                ) for content, metadata in zip(total_docs_content, total_docs_metadata)
            ]
    
            # Split the documents into chunks
            chunks = character_splitter.create_documents(
                texts=[doc.page_content for doc in chunks],
                metadatas=[doc.metadata for doc in chunks]
            )
    
        print(
            "SPLITTING DONE!, STARTING TO REMOVE DUPLICATE CHUNKS ..."
        )
    
        # remove duplicate chunks
        cleaned_chunks = remove_duplicate_chunks(chunks)
    
        # add the result to class variable
        self.total_chunks = cleaned_chunks
        
    def get_metrics(self):
        """Get some metrics about the chunks and print them in terminal."""
        
        #compute average chunk lenght in number of words by computing lenght of chunk.content and then summing them
        avg_chunk_length = sum([len(chunk.page_content.split()) for chunk in self.total_chunks]) / len(self.total_chunks)
        print("Average chunk length in number of words:", avg_chunk_length)

    @log_execution_time
    def load_vectordb(self):
        """
        Load the vectordb from the specified persist directory.

        Args:
            config (dict): The directory where the vectordb is persisted.

        Returns:
            vectordb: The loaded vectordb if it exists, otherwise None.
        """

        # if self.config["use_server"]:
        #     # print("Using the server to load the Qdrant database...")
        #     # from qdrant_client import QdrantClient

        #     # self.client = QdrantClient("localhost", port=6333)
        #     # qdrant = Qdrant.from_existing_collection(
        #     #     url=self.config["server_url"],
        #     #     embedding=self.dense_embedding_model,
        #     #     collection_name="qdrant_vectorstore",
        #     # )
        #     # self.vectordb = qdrant
        #     pass
        
        if os.path.exists(os.path.join(self.persist_directory, "collection")):
            print("Collection exists, loading the Qdrant database...")
            qdrant = Qdrant.from_existing_collection(
                path=self.persist_directory,
                embedding=self.dense_embedding_model,
                collection_name="qdrant_vectorstore",
            )
            self.vectordb = qdrant

        else:
            print("Collection does not exist, returning None...")
            self.vectordb = None
            
            
    def delete(self):
        """
        Deletes the vectorstore from the persist directory.
        """
        import shutil
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print("Vectorstore deleted successfully!")
            except PermissionError as e:
                print(f"PermissionError: {e}. Attempting to force delete.")
                self.force_delete(self.persist_directory)
        else:
            print("Vectorstore does not exist!")

    def force_delete(self, directory):
        """
        Forcefully deletes a directory by terminating processes that lock it.
        """
        import psutil
        import shutil

        for proc in psutil.process_iter():
            try:
                for file in proc.open_files():
                    if file.path.startswith(directory):
                        proc.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        shutil.rmtree(directory)
        print("Vectorstore forcefully deleted!")
        

    def add_documents_to_db_V2(self):
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

        logging.info(
            "TOTAL NUMBER OF CHUNKS GENERATED:" + str(len(self.total_chunks))
        )
        if len(self.total_chunks) == 0:
            logging.info(
                "No chunks to add to the vectorstore they are already there!"
            )

        logging.info("Starting to add documents to the dense vectorstore ...")
        if self.total_chunks:
            if self.vectordb is None:
                logging.info("Vectordb is None, creating a new one...")
                #print("IS using server:", self.config["use_server"])
                # if self.config["use_server"]:
                #     print("USING THE SERVER TO CREATE THE QDRANT DATABASE...")
                #     self.vectordb = QdrantVectorStore.from_documents(
                #     documents=self.total_chunks,
                #     embedding=self.dense_embedding_model,
                #     sparse_embedding=self.sparse_embedding_model,
                #     # url=self.config["server_url"],
                #     collection_name="qdrant_vectorstore",
                #     retrieval_mode=RetrievalMode.HYBRID,
                # )
                # else:
                self.vectordb = QdrantVectorStore.from_documents(
                    documents=self.total_chunks,
                    embedding=self.dense_embedding_model,
                    sparse_embedding=self.sparse_embedding_model,
                    path=self.persist_directory,
                    collection_name="qdrant_vectorstore",
                    retrieval_mode=RetrievalMode.HYBRID,
                )

            else:
                logging.info(
                    "Vectordb is not None, adding documents to the existing database..."
                )

                self.vectordb.add_documents(documents=self.total_chunks)

            logging.info("Qdrant database fully updated with new documents!")
        
        # #delete the instance or any other method to close the connection
        # self.vectordb.close()
        # #delete the instance
        # del self.vectordb
        
        

    def fill(self):
        #self.save_config_file()
        self.create_persist_directory() # create the persist directory if it does not exist
        self.get_log_file_path() # get the log file path or create it if it does not exist
        self.load_vectordb() # load the vector database if it exists, otherwise create a new one
        self.find_already_processed() # find the already processed documents
        self.process_documents() # process the documents
        print("Number of total documents:", len(self.total_documents))
        print("Number of total chunks:", len(self.total_chunks))
        self.filter_and_split_chunks() # filter and split the chunks
        print("Number of total documents:", len(self.total_documents))
        print("Number of total chunks:", len(self.total_chunks))
        self.get_metrics() # get some metrics about the chunks
        self.add_documents_to_db_V2()
        
    def get_chunks(self):
        """Return the chunks without pushing them to the vectorstore."""
        self.process_documents() # process the documents
        self.filter_and_split_chunks() # filter and split the chunks
        return self.total_chunks


def get_modif_date(path):
    # find modif date of the file in its metadata

    temps_modification = os.path.getmtime(path)

    # Convertir le temps en format de date lisible
    date_modification = datetime.fromtimestamp(temps_modification)

    return str(date_modification)


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


def remove_duplicate_chunks(chunks):
    """
    Remove duplicate chunks from a list of chunks and create Document objects, add chunk length and field to metadata, and clean the content.

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

        # Handle both forward and backward slashes and get the folder before the last part
        source_parts = source.replace("\\", "/").split("/")
        source_field = (
            source_parts[-2] if len(source_parts) > 1 else source_parts[0]
        )

        # obtain the modification date of the chunk based on the source
        modif_date = get_modif_date(chunk.metadata["source"])

        # print("FIELD:", source_field)
        # print("Source:", source)

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


if __name__ == "__main__":
    # Load the configuration file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Create a VectorAgent object
    agent = VectorAgent(default_config=config)

    agent.fill()

    # chunks = agent.get_chunks()
    # print("Number of chunks:", len(chunks))