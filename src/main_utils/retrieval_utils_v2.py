from functools import lru_cache
import torch
import yaml
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_qdrant import Qdrant, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
#from transformers import logger as transformers_logger

# custom imports
from src.main_utils.custom_langchain_componants import (
    CustomCrossEncoderReranker,
    TopKCompressor,
)
from src.main_utils.embedding_utils import (
    get_embedding_model,
    get_sparse_embedding_model,
)

from src.aux_utils.logging_utils import setup_logger,log_execution_time


# Load the environment variables (API keys, etc...)
load_dotenv()

logger = setup_logger(
    __name__,
    "query_database.log",
    log_format="%(asctime)s:%(levelname)s:%(message)s"
)


class RetrievalAgent:
    def __init__(self, default_config, config={}):
        """
        Initialize the retrieval utility with the given configuration.

        Args:
            default_config (dict): The default configuration dictionary.
            config (dict, optional): An optional configuration dictionary that 
                overrides the default configuration. Defaults to an empty dictionary.

        Attributes:
            config (dict): The combined configuration dictionary.
            embedding_model: The embedding model initialized based on the configuration.
            sparse_embedding_model: The sparse embedding model initialized based on the configuration.
            client (QdrantClient): The Qdrant client initialized with the persist directory from the configuration.
            raw_database: The existing Qdrant database retrieved using the configuration.
        """
        self.config = {**default_config, **config}
        self.embedding_model = get_embedding_model(
            model_name=self.config["embedding_model"]
        )
        self.sparse_embedding_model = get_sparse_embedding_model(
            model_name=self.config["sparse_embedding_model"]
        )
        # self.raw_database = self.get_existing_qdrant(self.config["persist_directory"], self.config["embedding_model"])
        self.client = QdrantClient(path=self.config["persist_directory"])
        self.raw_database = self.get_existing_qdrant_v3()

    def apply_field_filter_qdrant(
        self, search_kwargs, field_filter, field_filter_type
    ):
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

        def create_filter_condition(field, value, filter_type):
            if filter_type == "$ne":
                return {"must_not": [{"key": field, "match": {"value": value}}]}
            elif filter_type == "$eq":
                return {"must": [{"key": field, "match": {"value": value}}]}
            else:
                raise ValueError(f"Unsupported filter type: {filter_type}")

        if "filter" not in search_kwargs:
            search_kwargs["filter"] = {}

        existing_filter = search_kwargs["filter"]

        if isinstance(field_filter, list) and len(field_filter) > 1:
            conditions = [
                create_filter_condition("field", field, field_filter_type)
                for field in field_filter
            ]
            if field_filter_type == "$ne":
                existing_filter.setdefault("should", []).extend(conditions)
            elif field_filter_type == "$eq":
                existing_filter.setdefault("must", []).extend(conditions)
        else:
            field = (
                field_filter
                if isinstance(field_filter, str)
                else field_filter[0]
            )
            condition = create_filter_condition(
                "field", field, field_filter_type
            )
            if field_filter_type == "$ne":
                existing_filter.setdefault("must_not", []).extend(
                    condition["must_not"]
                )
            elif field_filter_type == "$eq":
                existing_filter.setdefault("must", []).extend(condition["must"])

        print(
            "Searching for the following fields:",
            field_filter,
            "with filter type:",
            field_filter_type,
        )
        return search_kwargs

    def get_filtering_kwargs_qdrant(
        self,
        source_filter,
        source_filter_type,
        field_filter,
        field_filter_type,
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
            return qdrant_models.FieldCondition(
                key=field, match=qdrant_models.MatchValue(value=value)
            )

        conditions_should = []
        conditions_must = []
        conditions_not = []

        if source_filter:
            for source in source_filter:
                condition = create_filter_condition("metadata.source", source)
                if source_filter_type == "$eq":
                    conditions_should.append(condition)
                else:
                    conditions_not.append(condition)

        if field_filter:
            for field in field_filter:
                condition = create_filter_condition("metadata.field", field)
                if field_filter_type == "$eq":
                    conditions_should.append(condition)
                else:
                    conditions_not.append(condition)

        if length_threshold is not None:
            length_condition = qdrant_models.FieldCondition(
                key="metadata.chunk_length",
                range=qdrant_models.Range(gte=float(length_threshold)),
            )
            if not field_filter:
                conditions_should.append(length_condition)
            else:
                conditions_must.append(length_condition)

        search_kwargs["filter"] = qdrant_models.Filter(
            should=conditions_should,
            must=conditions_must,
            must_not=conditions_not,
        )

        return search_kwargs

    @log_execution_time
    def apply_advanced_hybrid_search_v3(
        self, base_retriever, nb_chunks, query, search_kwargs
    ):
        """
        Applies advanced hybrid search by filtering and retrieving documents based on various criteria (V3 is using a unique vectorstore instead of two).

        Args:
            base_retriever (Retriever): The base retriever used for retrieval.
            nb_chunks (int): The number of chunks to retrieve.
            query (str): The query to search for.
            search_kwargs (dict): The search keyword arguments.

        Returns:
            Retriever: The hybrid retriever with the filtered and retrieved documents.
        """

        base_retriever = self.raw_database.as_retriever(
            search_kwargs=search_kwargs, search_type=self.config["search_type"]
        )

        return base_retriever

    @lru_cache(maxsize=None)
    def load_reranker(self, model_name, device="cuda", show_progress=False):
        """
        Load the reranker model based on the given model name and device.

        Args:
            model_name (str): The name of the model.
            device (str): The device to load the model on. Default is "cuda".
            show_progress (bool): Whether to show progress bars during model loading. Default is False.

        Returns:
            Reranker: The reranker model object.
        """
        # if not show_progress:
        #     transformers_logger.set_verbosity_error()
        #     logger.getLogger("transformers").setLevel(logger.ERROR)

        model_kwargs = (
            {
                "automodel_args": {"torch_dtype": torch.float16},
                "device": device,
                "trust_remote_code": True,
                "max_length": 1024,
            }
            if "jina" in model_name.lower()
            else {"device": device}
        )

        return HuggingFaceCrossEncoder(
            model_name=model_name, model_kwargs=model_kwargs
        )

    @log_execution_time
    def apply_reranker(self, query, base_retriever):
        """
        Applies a reranker to enhance precision and compresses documents before reranking.

        Args:
            query (str): The query to retrieve relevant documents.
            base_retriever: The base retriever to use.

        Returns:
            compressed_docs: The compressed documents.
        """

        reranker = self.load_reranker(self.config["reranker_model"])

        logger.info("RERANKER LOADED !")

        intelligent_compression = self.config["llm_token_target"] != 0

        reranker_compressor = CustomCrossEncoderReranker(
            model=reranker,
            top_n=self.config["nb_rerank"],
            use_autocut=self.config["use_autocut"],
            autocut_beta=self.config["autocut_beta"],
            intelligent_compression=intelligent_compression,
            token_target=self.config["llm_token_target"],
        )

        intelligent_compression = self.config["reranker_token_target"] not in [
            0,
            None,
        ]

        # Apply the top-k compressor
        top_k_compressor = TopKCompressor(
            k=self.config["nb_chunks"],
            intelligent_compression=intelligent_compression,
            token_target=self.config["reranker_token_target"],
        )

        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[top_k_compressor, reranker_compressor]
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever,
        )

        compressed_docs = compression_retriever.invoke(query)
        
        return compressed_docs

    # @lru_cache(maxsize=None)
    def get_existing_qdrant(self, persist_directory, embedding_model_name):
        """
        Get an existing Qdrant database from the specified directory.

        Args:
            persist_directory (str): The directory where the Qdrant database is stored.
            embedding_model_name (str): The name of the embedding model used in the database.

        Returns:
            Qdrant: The existing Qdrant database.
        """
        embedding_model = get_embedding_model(model_name=embedding_model_name)
        return Qdrant.from_existing_collection(
            path=persist_directory,
            embedding=embedding_model,
            collection_name="qdrant_vectorstore",
        )

    def get_existing_qdrant_v3(self):
        """
        Get an existing Qdrant database using QdrantClient.

        Args:
            persist_directory (str): The directory where the Qdrant database is stored.
            embedding_model_name (str): The name of the embedding model used in the database.

        Returns:
            QdrantVectorStore: The existing Qdrant database if collection exists, otherwise None.
        """
       

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if "qdrant_vectorstore" in [c.name for c in collections]:
                logger.info(
                    "Collection exists, loading the Qdrant database..."
                )

                vectordb = QdrantVectorStore(
                    client=self.client,
                    collection_name="qdrant_vectorstore",
                    sparse_vector_name="sparse",
                    embedding=self.embedding_model,
                    sparse_embedding=self.sparse_embedding_model,
                    retrieval_mode=RetrievalMode.HYBRID,
                )
                logger.info("Qdrant database loaded successfully")
                return vectordb

            else:
                logger.info("Collection does not exist, returning None...")
                return None

        except Exception as e:
            logger.error(f"Error loading Qdrant database: {str(e)}")
            raise

    @log_execution_time
    def query_database(self, query):
        """
        Query the database using the specified configuration.

        Args:
            query (str): The query string.
            default_config (dict): The default configuration.
            config (dict, optional): The additional configuration. Defaults to {}.

        Returns:
            list: The list of compressed documents. format: [Document, Document, ...]
        """

        logger.info("Trying to load the qdrant database...")

        logger.info("USING QDRANT SEARH KWARGS !")
        search_kwargs = self.get_filtering_kwargs_qdrant(
            source_filter=self.config["source_filter"],
            source_filter_type=self.config["source_filter_type"],
            field_filter=self.config["field_filter"],
            field_filter_type=self.config["field_filter_type"],
            length_threshold=self.config["length_threshold"],
        )

        search_kwargs["k"] = self.config["nb_chunks"]

        logger.info("QUERY: %s", query)
        logger.info("USED SEARCH KWARGS: %s", search_kwargs)

        if self.config["hybrid_search"]:
            search_kwargs["where_document"] = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="word_filter",
                        match=qdrant_models.MatchValue(
                            value=self.config["word_filter"]
                        ),
                    )
                ]
            )

        base_retriever = self.raw_database.as_retriever(
            search_kwargs=search_kwargs, search_type=self.config["search_type"]
        )

        if self.config["use_multi_query"]:
            base_retriever = self.apply_multi_query_retriever(
                base_retriever, self.config["nb_chunks"]
            )

        if self.config["advanced_hybrid_search"]:
            base_retriever = self.apply_advanced_hybrid_search_v3(
                base_retriever=base_retriever,
                nb_chunks=self.config["nb_chunks"],
                query=query,
                search_kwargs=search_kwargs,
            )

        if self.config["use_reranker"]:
            compressed_docs = self.apply_reranker(
                query=query, base_retriever=base_retriever
            )
        else:
            compressed_docs = base_retriever.get_relevant_documents(query=query)

        logger.info(
            "FIELDS FOUND: %s", [k.metadata["field"] for k in compressed_docs]
        )
        return compressed_docs


if __name__ == "__main__":
    with open("config/test_config.yaml") as f:
        config = yaml.safe_load(f)

    agent = RetrievalAgent(config)

    # Query the database
    query = "How to write a good resume?"
    compressed_docs = agent.query_database(query, config)
    print("Compressed documents:", compressed_docs)
