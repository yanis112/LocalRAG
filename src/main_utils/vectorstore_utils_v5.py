### import argparse
import json
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from rapidfuzz import process, fuzz

from src.aux_utils.logging_utils import log_execution_time, setup_logger

# custom imports
from src.main_utils.embedding_utils import (
    get_embedding_model,
    get_sparse_embedding_model,
)
from src.main_utils.utils import (
    text_preprocessing,
)

# Load the environment variables (API keys, etc...)
load_dotenv()

logger = setup_logger(
    __name__,
    "vectorstore_utils.log",
    log_format="%(asctime)s:%(levelname)s:%(message)s",
)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf._reader")

def trouver_correspondance_approximative(document, motif, seuil=95):
    """
    Recherche une correspondance approximative du motif dans le document.

    :param document: Le texte dans lequel effectuer la recherche.
    :param motif: La chaîne à rechercher.
    :param seuil: Le score minimal de similarité pour considérer une correspondance (0 à 100).
    :return: Un tuple (indice_debut, indice_fin, score) ou None si aucune correspondance n'est trouvée.
    """
    resultats = process.extract(
        motif, [document], scorer=fuzz.partial_ratio, score_cutoff=seuil
    )

    if resultats:
        meilleur_resultat = resultats[0]
        score = meilleur_resultat[1]
        indice_debut = meilleur_resultat[2]
        indice_fin = indice_debut + len(motif)
        return indice_debut, indice_fin, score

    return None

class VectorAgent:
    """
    A class to manage a vectorstore for a RAG pipeline.

    Attributes:
        config (dict): Configuration settings.
        persist_directory (str): Path to the directory where the vectorstore is stored.
        process_log_file (str): Name of the log file for processed documents.
        log_file_path (str): Full path to the log file.
        already_processed_docs (set): Set of already processed document names.
        processed_docs_data (dict): Dictionary mapping document names to their subfolder path and ID.
        allowed_formats (list): List of allowed file formats.
        dense_embedding_model: The dense embedding model.
        chunking_embedding_model: The embedding model used for semantic chunking.
        sparse_embedding_model: The sparse embedding model.
        client (QdrantClient): The Qdrant client.
        collection_name (str): Name of the Qdrant collection.
        total_chunks (list): List of all document chunks.
        total_documents (list): List of all processed documents.
        vectordb (QdrantVectorStore): The Qdrant vectorstore.
    """

    def __init__(self, default_config, config=None, qdrant_client=None):
        """
        Initializes the VectorAgent.
        """
        if config is None:
            config = {}
        self.config = {**default_config, **config}
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
        self.processed_docs_data = {}
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

        self.client = (
            QdrantClient(path=self.persist_directory)
            if qdrant_client is None
            else qdrant_client
        )
        self.collection_name = self.config["collection_name"]

        # storage variables
        self.total_chunks = []
        self.total_documents = []
        # initialize the processed docs dictionary to store the mapping path-id
        self.processed_docs_data = {}

    # --- File Management Functions ---

    def create_persist_directory(self):
        """
        Creates the persist directory if it does not already exist.
        """
        if not os.path.exists(self.persist_directory):
            logger.info("Persist directory does not exist!, creating it ...")
            os.makedirs(self.persist_directory)
            logger.info("Persist directory created successfully!")

    def get_log_file_path(self):
        """
        Gets the path of the log file, creating it if it does not exist.
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
        Saves the configuration file used to create the vectorstore.
        """
        try:
            with open(
                os.path.join(self.persist_directory, "building_config.yaml"), "w"
            ) as file:
                yaml.dump(self.config, file)
        except OSError as e:
            logger.error(f"Error saving config file: {e}")

    def load_processed_documents(self):
        """
        Loads the list of already processed documents from the log file.
        """
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as file:
                try:
                    self.processed_docs_data = json.load(file)
                except json.JSONDecodeError:
                    self.processed_docs_data = {}

        except FileNotFoundError:
            self.processed_docs_data = {}

        self.already_processed_docs = set(self.processed_docs_data.keys())
        print(
            f"Number of already processed documents: {len(self.already_processed_docs)}"
        )

    def save_processed_docs(self):
        """
        Saves the processed documents mapping to the log file.
        """
        try:
            with open(self.log_file_path, "w", encoding="utf-8") as file:
                json.dump(self.processed_docs_data, file, indent=4)
        except OSError as e:
            logger.error(f"Error saving processed documents to log file: {e}")

    def get_pending_files(self):
        """
        Retrieves a set of pending files that need to be processed.

        Returns:
            set: A set of tuples, where each tuple contains the file name and its full path.
        """
        pending_files = set()
        for root, _, files in os.walk(self.path):
            for name in files:
                if (
                    name.split(".")[-1] in self.allowed_formats
                    and name not in self.already_processed_docs
                    and name != "meta.json"
                ):
                    pending_files.add((name, os.path.join(root, name)))
        return pending_files

    # --- Document Processing Functions ---

    def process_pending_documents(self):
        """
        Processes all new documents that haven't been processed before.
        """
        pending_files = self.get_pending_files()
        if not pending_files:
            print("No new files to process")
            return

        with tqdm(total=len(pending_files), desc="Processing new files") as pbar:
            for name, full_path in pending_files:
                doc = self.load_document(name, full_path)
                if doc:
                    self.total_documents.append(doc)
                    subfolder_path = os.path.dirname(full_path)
                    self.processed_docs_data[name] = {
                        "subfolder_path": str(Path(subfolder_path).as_posix())
                    }
                    self.already_processed_docs.add(name)
                pbar.update()

    def load_document(self, name, full_path):
        """
        Loads a document using the appropriate loader based on its file extension.

        Args:
            name (str): The name of the document.
            full_path (str): The full path to the document.

        Returns:
            object: The loaded document if successful, otherwise None.
        """
        try:
            loader = self._get_document_loader(name, full_path)
            doc = loader.load()
            return doc
        except Exception as e:
            logger.error(f"Error processing {name}: {traceback.format_exc()}")
            return None

    def _get_document_loader(self, name, full_path):
        """
        Returns the appropriate document loader based on the file extension.

        Args:
            name (str): The name of the file.
            full_path (str): The full path to the file.

        Returns:
            DocumentLoader: The appropriate document loader.

        Raises:
            ValueError: If the file format is not supported.
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
            raise ValueError(f"Unsupported file format: {ext} for file: {name}")

        return loader

    @log_execution_time
    def split_documents_into_chunks(self):
        """
        Splits documents into chunks based on the specified splitting method.
        """
        print("STARTING TO SPLIT THE DOCUMENTS INTO CHUNKS ...")

        docs_to_split = []
        docs_not_to_split = []
        for doc in self.total_documents:
            if "sheets" in doc[0].metadata["source"]:
                print("Document not to split Metadata:", doc[0].metadata)
                print("Document not to split:", doc[0].metadata["source"])
                docs_not_to_split.append(doc)
            else:
                docs_to_split.append(doc)

        chunks_not_to_split = [
            Document(page_content=doc[0].page_content, metadata=doc[0].metadata)
            for doc in docs_not_to_split
        ]
        
        if self.config["splitting_method"] == "semantic":
            semantic_splitter = SemanticChunker(
                embeddings=self.chunking_embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=self.config["semantic_threshold"],
            )
            total_docs_content = [doc[0].page_content for doc in docs_to_split]
            total_docs_metadata = [doc[0].metadata for doc in docs_to_split]
            
            # Create temporary documents for semantic splitting
            temp_chunks = semantic_splitter.create_documents(
                texts=total_docs_content, metadatas=total_docs_metadata
            )
            
            chunks_to_split = []
            for i, temp_chunk in enumerate(temp_chunks):
                # Find the index of original document
                original_doc_indices = [
                    idx for idx, doc in enumerate(docs_to_split) 
                    if doc[0].metadata['source'] == temp_chunk.metadata['source']
                ]
                
                if not original_doc_indices:
                    print(f"Original document not found for chunk {i}")
                    start_index, end_index = -1, -1
                else:
                    original_doc_index = original_doc_indices[0]
                    original_doc_content = total_docs_content[original_doc_index]
                    
                    # Use string find to locate the chunk in the original document
                    start_index = original_doc_content.find(temp_chunk.page_content)
                    if start_index != -1:
                        end_index = start_index + len(temp_chunk.page_content)
                    else:
                        start_index, end_index = -1, -1
                        print(f"Chunk content not found in original document for chunk {i}")

                # Clean the chunk content
                cleaned_content = text_preprocessing(temp_chunk.page_content)

                chunks_to_split.append(
                    Document(
                        page_content=cleaned_content,
                        metadata={
                            **temp_chunk.metadata,
                            "start_index": start_index,
                            "end_index": end_index
                        }
                    )
                )
            
        else:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            chunk_size = self.config["chunk_size"]
            chunk_overlap = self.config["chunk_overlap"]
            character_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            total_docs_content = [doc[0].page_content for doc in docs_to_split]
            total_docs_metadata = [doc[0].metadata for doc in docs_to_split]
            
            # Create temporary documents for index calculation
            temp_docs = [
                Document(page_content=content, metadata=metadata)
                for content, metadata in zip(total_docs_content, total_docs_metadata)
            ]
            
            # Split documents into chunks
            temp_chunks = character_splitter.split_documents(temp_docs)
            
            chunks_to_split = []
            for i, temp_chunk in enumerate(temp_chunks):
                # Find the index of original document
                original_doc_indices = [
                    idx for idx, doc in enumerate(docs_to_split) 
                    if doc[0].metadata['source'] == temp_chunk.metadata['source']
                ]

                if not original_doc_indices:
                    print(f"Original document not found for chunk {i}")
                    start_index, end_index = -1, -1
                else:
                    original_doc_index = original_doc_indices[0]
                    original_doc_content = total_docs_content[original_doc_index]
                    
                    # Use string find to locate the chunk in the original document
                    start_index = original_doc_content.find(temp_chunk.page_content)
                    if start_index != -1:
                        end_index = start_index + len(temp_chunk.page_content)
                    else:
                        start_index, end_index = -1, -1
                        print(f"Chunk content not found in original document for chunk {i}")
                
                # Clean the chunk content
                cleaned_content = text_preprocessing(temp_chunk.page_content)

                chunks_to_split.append(
                    Document(
                        page_content=cleaned_content,
                        metadata={
                            **temp_chunk.metadata,
                            "start_index": start_index,
                            "end_index": end_index
                        }
                    )
                )

        print("SPLITTING DONE!, STARTING TO REMOVE DUPLICATE CHUNKS ...")

        chunks = chunks_to_split + chunks_not_to_split
        self.total_chunks = self._create_document_objects(chunks)

    def _create_document_objects(self, chunks):
        """
        Creates Document objects from a list of chunks, removing duplicates and adding metadata, 
        including start and end character indices.

        Args:
            chunks (list): A list of Chunk objects.
            original_docs (list): A list of the original, unsplit Document objects.

        Returns:
            list: A list of Document objects with added metadata.
        """
        unique_chunks = []
        seen_contents = set()
        for chunk in chunks:
            if chunk.page_content not in seen_contents:
                seen_contents.add(chunk.page_content)
                unique_chunks.append(chunk)

        document_objects = []
        for chunk in tqdm(unique_chunks, desc="Creating Document objects"):
            source = truncate_path_to_data(chunk.metadata["source"])
            source_parts = source.replace("\\", "/").split("/")
            source_field = (
                source_parts[-2] if len(source_parts) > 1 else source_parts[0]
            )
            modif_date = get_modif_date(chunk.metadata["source"])

            document_objects.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "source": source,
                        "chunk_length": len(chunk.page_content.split()),
                        "field": source_field,
                        "modif_date": modif_date,
                        "start_index": chunk.metadata.get("start_index", -1),
                        "end_index": chunk.metadata.get("end_index", -1),
                    },
                )
            )
        return document_objects

    def compute_and_print_metrics(self):
        """
        Computes and prints metrics about the chunks.
        """
        if not self.total_chunks:
            print("No chunks to analyze")
            return

        try:
            valid_lengths = [
                len(chunk.page_content.split())
                for chunk in self.total_chunks
                if chunk and chunk.page_content and chunk.page_content.strip()
            ]

            if not valid_lengths:
                print("No valid chunks with content found")
                return

            avg_chunk_length = sum(valid_lengths) / len(valid_lengths)
            print(f"Average chunk length in number of words: {avg_chunk_length:.2f}")
            print(f"Total valid chunks analyzed: {len(valid_lengths)}")

        except Exception as e:
            print(f"Error computing metrics: {str(e)}")

    # --- Vectorstore Management Functions ---

    @log_execution_time
    def load_vectorstore(self):
        """
        Loads the vectorstore from the persist directory if the collection exists.

        Returns:
            QdrantVectorStore: The loaded vectorstore if collection exists, otherwise None.
        """
        try:
            collections = self.client.get_collections().collections
            if self.collection_name in [c.name for c in collections]:
                logger.info("Collection exists, loading the Qdrant database...")
                self.vectordb = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.dense_embedding_model,
                    sparse_embedding=self.sparse_embedding_model,
                    sparse_vector_name="sparse",
                    retrieval_mode=RetrievalMode.HYBRID,
                )
                logger.info("Qdrant database loaded successfully")
            else:
                logger.info("Collection does not exist, returning None...")
                self.vectordb = None
        except Exception as e:
            logger.error(f"Error loading vectorstore: {str(e)}")
            self.vectordb = None
            raise

        return self.vectordb

    def delete(self, ids=None, paths=None, folders=None):
        """
        Deletes documents from the vectorstore based on file names, file paths, or folders.

        Args:
            ids (list, optional): A list of document IDs to delete. Defaults to None.
            paths (list, optional): A list of document paths relative to root to delete. Defaults to None.
            folders (list, optional): A list of folder paths to delete documents from. Defaults to None.
        """
        ids_to_delete = set()
        if ids:
            ids_to_delete.update(ids)
        if paths:
            for path in paths:
                for doc_name, data in self.processed_docs_data.items():
                    if data["subfolder_path"] == path:
                        ids_to_delete.add(data["id"])
        if folders:
            for folder in folders:
                for doc_name, data in self.processed_docs_data.items():
                    print("subfolder_data:", data["subfolder_path"])
                    print("folder:", folder)
                    if data["subfolder_path"] == folder:
                        ids_to_delete.add(data["id"])

        ids_to_delete = list(ids_to_delete)
        if not ids_to_delete:
            print("No documents to delete.")
            return

        try:
            docs_to_remove_physically = [
                key
                for key, value in self.processed_docs_data.items()
                if value["id"] in ids_to_delete
            ]

            result = self.vectordb.delete(ids=ids_to_delete)

            for doc_name in docs_to_remove_physically:
                file_path = os.path.join(
                    self.processed_docs_data[doc_name]["subfolder_path"], doc_name
                )
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error while deleting physical file: {file_path} , {e}")

            self.processed_docs_data = {
                k: v
                for k, v in self.processed_docs_data.items()
                if v["id"] not in ids_to_delete
            }
            self.save_processed_docs()
            self.already_processed_docs = set(self.processed_docs_data.keys())

            print(f"Successfully deleted {len(ids_to_delete)} documents.")
            return result
        except Exception as e:
            print(f"Error details: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())

    def delete_all(self):
        """
        Deletes the vectorstore from the persist directory.
        """
        if os.path.exists(self.persist_directory):
            try:
                import shutil

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

        Args:
            directory (str): The directory to delete.
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

    @log_execution_time
    def add_documents_to_vectorstore(self):
        """
        Adds documents to the vector database.
        """
        logger.info(f"TOTAL NUMBER OF CHUNKS GENERATED: {len(self.total_chunks)}")
        if len(self.total_chunks) == 0:
            logger.info("No chunks to add to the vectorstore - they are already there!")
            return

        logger.info("Starting to add documents to the vectorstore...")

        try:
            if self.total_chunks:
                if self.vectordb is None:
                    logger.info("Vectordb is None, checking for existing collection...")

                    if not self.client.collection_exists(self.collection_name):
                        from qdrant_client.http.models import (
                            Distance,
                            VectorParams,
                        )

                        logger.info("Collection does not exist, creating a new one...")
                        sparse_vector_name = "sparse"

                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(
                                size=self.config["dense_embedding_size"],
                                distance=Distance.COSINE,
                            ),
                            sparse_vectors_config={
                                sparse_vector_name: models.SparseVectorParams(
                                    index=models.SparseIndexParams(on_disk=True)
                                )
                            },
                        )

                    self.vectordb = QdrantVectorStore(
                        client=self.client,
                        collection_name=self.collection_name,
                        embedding=self.dense_embedding_model,
                        sparse_embedding=self.sparse_embedding_model,
                        sparse_vector_name="sparse",
                        retrieval_mode=RetrievalMode.HYBRID,
                    )

                logger.info("Adding documents to existing vectorstore...")
                id_list = self.vectordb.add_documents(documents=self.total_chunks)

                for i, chunk in enumerate(self.total_chunks):
                    file_name = os.path.basename(chunk.metadata["source"])
                    if file_name in self.processed_docs_data:
                        self.processed_docs_data[file_name]["id"] = id_list[i]
                    else:
                        self.processed_docs_data[file_name] = {"id": id_list[i]}

                logger.info("Qdrant database successfully updated with new documents!")

        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {str(e)}")
            raise e

    # --- Main Functions ---

    def build_vectorstore(self):
        """
        Builds the vectorstore from scratch.
        """
        self.create_persist_directory()
        self.get_log_file_path()
        self.load_vectorstore()
        self.load_processed_documents()
        print(
            "Number of already processed documents:", len(self.already_processed_docs)
        )
        self.process_pending_documents()
        self.split_documents_into_chunks()
        print(
            "Number of total documents currently processed:",
            len(self.total_documents),
        )
        print(
            "Number of total chunks currently processed:",
            len(self.total_chunks),
        )
        self.compute_and_print_metrics()
        self.add_documents_to_vectorstore()
        self.save_processed_docs()

    def get_chunks(self):
        """
        Returns the chunks without pushing them to the vectorstore.
        """
        self.process_pending_documents()
        self.split_documents_into_chunks()
        return self.total_chunks

# --- Helper Functions ---

def get_modif_date(path):
    """
    Gets the modification date of a file.

    Args:
        path (str): The path to the file.

    Returns:
        str: The modification date as a string.
    """
    try:
        modification_time = os.path.getmtime(path)
        modification_date = datetime.fromtimestamp(modification_time)
        return str(modification_date)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except OSError as e:
        logger.error(f"Error getting modification date for {path}: {e}")
        return None

def truncate_path_to_data(path):
    """
    Truncates the given path to start from '/data' if '/data' is present in the path.

    Args:
        path (str): The original file path.

    Returns:
        str: The truncated or original path.
    """
    if "data/" in path:
        data_index = path.index("data/")
        return path[data_index:]
    else:
        return path

if __name__ == "__main__":
    # load the config file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # initialize the VectorAgent
    agent = VectorAgent(default_config=config)

    # try to build the vectorstore
    try:
        agent.build_vectorstore()
        print("Database built successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

# parser = argparse.ArgumentParser(
#     description="Build or manage a vector database from a config file."
# )
# parser.add_argument(
#     "--config",
#     type=str,
#     default="config/config.yaml",
#     help="Path to config file (default: config/config.yaml)",
# )
# parser.add_argument(
#     "--delete_ids",
#     type=str,
#     nargs="*",
#     help="List of document ids to delete",
# )
# parser.add_argument(
#     "--delete_paths",
#     type=str,
#     nargs="*",
#     help="List of document paths to delete",
# )
# parser.add_argument(
#     "--delete_folders",
#     type=str,
#     nargs="*",
#     help="List of folders to delete",
# )
# parser.add_argument(
#     "--delete_all",
#     action="store_true",
#     help="Delete all data in the vectorstore",
# )

# args = parser.parse_args()
# config_path = Path(args.config)

# try:
#     if not config_path.exists():
#         print(f"Error: Config file not found at {config_path}")
#         sys.exit(1)

#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)

#     print(f"Loading config from: {config_path}")
#     agent = VectorAgent(default_config=config)

#     if args.delete_ids or args.delete_paths or args.delete_folders:
#         ids_to_delete = [id for id in args.delete_ids] if args.delete_ids else None
#         result = agent.delete(
#             ids=ids_to_delete, paths=args.delete_paths, folders=args.delete_folders
#         )
#         if result:
#             print("Successfully deleted the selected files from the vectorstore")
#         else:
#             print("No files were deleted")
#         sys.exit(0)
#     elif args.delete_all:
#         agent.delete_all()
#         sys.exit(0)

#     print("Building vector database...")
#     agent.build_vectorstore()
#     print("Database built successfully!")

# except Exception as e:
#     print(f"Error: {str(e)}")
#     sys.exit(1) ###