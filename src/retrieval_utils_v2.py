import logging
import time
from functools import lru_cache

import streamlit as st
import torch
import yaml
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_qdrant import Qdrant
from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from transformers import logging as transformers_logging

# custom imports
from src.custom_langchain_componants import *  # noqa:
from src.embedding_model import get_embedding_model
from src.LLM import CustomChatModel
# from src.query_routing_utils import QueryRouter
from src.utils import get_vram_logging, log_execution_time

# Load the environment variables (API keys, etc...)
load_dotenv()

class RetrievalAgent:
    def __init__(self, config):
        self.config = config
        #self.client = self.load_client(config["persist_directory"])
        self.embedding_model = get_embedding_model(model_name=config["embedding_model"])
        self.raw_database = self.get_existing_qdrant(config["persist_directory"], config["embedding_model"])

   
    def apply_field_filter_qdrant(self,search_kwargs, field_filter, field_filter_type):
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
            conditions = [create_filter_condition("field", field, field_filter_type) for field in field_filter]
            if field_filter_type == "$ne":
                existing_filter.setdefault("should", []).extend(conditions)
            elif field_filter_type == "$eq":
                existing_filter.setdefault("must", []).extend(conditions)
        else:
            field = field_filter if isinstance(field_filter, str) else field_filter[0]
            condition = create_filter_condition("field", field, field_filter_type)
            if field_filter_type == "$ne":
                existing_filter.setdefault("must_not", []).extend(condition["must_not"])
            elif field_filter_type == "$eq":
                existing_filter.setdefault("must", []).extend(condition["must"])

        print("Searching for the following fields:", field_filter, "with filter type:", field_filter_type)
        return search_kwargs


    def get_filtering_kwargs_qdrant(self,source_filter, source_filter_type, field_filter, field_filter_type, length_threshold=None):
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
            return qdrant_models.FieldCondition(key=field, match=qdrant_models.MatchValue(value=value))

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
            should=conditions_should, must=conditions_must, must_not=conditions_not
        )

        return search_kwargs

    @log_execution_time
    def apply_multi_query_retriever(self, base_retriever, nb_chunks):
        """
        Applies a multi-query retriever to the given base retriever.

        Args:
            base_retriever: The base retriever object.
            nb_chunks: The number of chunks to generate queries from the main query.

        Returns:
            vector_db_multi: The multi-query retriever object.
        """
        chat_model_object = CustomChatModel(llm_name="llama3", llm_provider="ollama")
        llm = chat_model_object.chat_model
        vector_db_multi = CustomMultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            include_original=True,
            top_k=nb_chunks,
        )
        return vector_db_multi

  
    #@lru_cache(maxsize=None)
    def load_client(self,custom_persist):
        """
        Load the Qdrant client.

        Args:
            custom_persist (str): The custom persist directory.

        Returns:
            QdrantClient: The Qdrant client object.
        """
        return QdrantClient(path=custom_persist)

  
    @lru_cache(maxsize=None)
    def load_router(self):
        """
        Load the query router object.

        Returns:
            QueryRouter: The query router object.
        """
        query_router = QueryRouter()
        query_router.load()
        return query_router

    @log_execution_time
    def apply_advanced_hybrid_search_v3(self, base_retriever, nb_chunks, query, search_kwargs):
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
        config = self.config

        if config["enable_routing"]:
            query_router = self.load_router()
            label = query_router.predict_label(query)
            alpha = query_router.predict_alpha(query)
            try:
                st.toast("Query type: " + str(label), icon="🔍")
            except:
                pass
        else:
            label = "NO LABEL, USING DEFAULT ALPHA"
            alpha = config["alpha"]

       

        base_retriever = self.raw_database.as_retriever(
            search_kwargs=search_kwargs, search_type=config["search_type"]
        )

    
        return base_retriever


    @lru_cache(maxsize=None)
    def load_reranker(self,model_name, device="cuda", show_progress=False):
        """
        Load the reranker model based on the given model name and device.

        Args:
            model_name (str): The name of the model.
            device (str): The device to load the model on. Default is "cuda".
            show_progress (bool): Whether to show progress bars during model loading. Default is False.

        Returns:
            Reranker: The reranker model object.
        """
        if not show_progress:
            transformers_logging.set_verbosity_error()
            logging.getLogger("transformers").setLevel(logging.ERROR)

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

        return HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)

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
        config = self.config

        reranker = self.load_reranker(config["reranker_model"])
       
        logging.info("RERANKER LOADED !")

        intelligent_compression = config["llm_token_target"] != 0

        reranker_compressor = CustomCrossEncoderReranker(
            model=reranker,
            top_n=config["nb_rerank"],
            use_autocut=config["use_autocut"],
            autocut_beta=config["autocut_beta"],
            intelligent_compression=intelligent_compression,
            token_target=config["llm_token_target"],
        )

        intelligent_compression = config["reranker_token_target"] not in [0, None]

        top_k_compressor = TopKCompressor(
            k=config["nb_chunks"],
            intelligent_compression=intelligent_compression,
            token_target=config["reranker_token_target"],
        )
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[top_k_compressor, reranker_compressor]
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever,
        )

        compressed_docs = compression_retriever.get_relevant_documents(query=query)
        return compressed_docs

   
    #@lru_cache(maxsize=None)
    def get_existing_qdrant(self,persist_directory, embedding_model_name):
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

    @traceable
    @log_execution_time
    def query_database(self, query, default_config, config={}):
        """
        Query the database using the specified configuration.

        Args:
            query (str): The query string.
            default_config (dict): The default configuration.
            config (dict, optional): The additional configuration. Defaults to {}.

        Returns:
            list: The list of compressed documents. format: [Document, Document, ...]
        """
        config = {**default_config, **config}

        logging.info("Trying to load the qdrant database...")

        logging.info("USING QDRANT SEARH KWARGS !")
        search_kwargs = self.get_filtering_kwargs_qdrant(
            source_filter=config["source_filter"],
            source_filter_type=config["source_filter_type"],
            field_filter=config["field_filter"],
            field_filter_type=config["field_filter_type"],
            length_threshold=config["length_threshold"],
        )

        search_kwargs["k"] = config["nb_chunks"]

        logging.info("QUERY: %s", query)
        logging.info("USED SEARCH KWARGS: %s", search_kwargs)

        if config["hybrid_search"]:
            search_kwargs["where_document"] = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="word_filter",
                        match=qdrant_models.MatchValue(value=config["word_filter"]),
                    )
                ]
            )

        base_retriever = self.raw_database.as_retriever(
            search_kwargs=search_kwargs, search_type=config["search_type"]
        )

        if config["use_multi_query"]:
            base_retriever = self.apply_multi_query_retriever(base_retriever, config["nb_chunks"])

        if config["advanced_hybrid_search"]:
            base_retriever = self.apply_advanced_hybrid_search_v3(
                base_retriever=base_retriever,
                nb_chunks=config["nb_chunks"],
                query=query,
                search_kwargs=search_kwargs,
            )

        if config["use_reranker"]:
            compressed_docs = self.apply_reranker(query=query, base_retriever=base_retriever)
        else:
            compressed_docs = base_retriever.get_relevant_documents(query=query)

        logging.info("FIELDS FOUND: %s", [k.metadata["field"] for k in compressed_docs])
        return compressed_docs

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
        
    agent = RetrievalAgent(config)
    
    
    # Query the database
    query = "How to write a good resume?"
    compressed_docs = agent.query_database(query, config)
    print("Compressed documents:", compressed_docs)