import logging
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch

# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    OllamaEmbeddings,
)
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


@lru_cache(maxsize=None)
def get_embedding_model(model_name, show_progress=False):
    """
    Returns an embedding model based on the given model name.
    Args:
        model_name (str): The name of the model.
        show_progress (bool, optional): Whether to show progress or not. Defaults to False.
    Returns:
        embed (object): The embedding model object.
    Raises:
        None
    Examples:
        >>> model = get_embedding_model("bge_model")
        >>> print(model)
        <HuggingFaceBgeEmbeddings object at 0x7f9a2e3e4a90>
    """
    
    # Désactiver les logs de transformers si show_progress est False
    if not show_progress:
        logging.getLogger("transformers").setLevel(logging.ERROR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}

    if "bge" in model_name and not "ollama" in model_name:
        encode_kwargs = {
            "normalize_embeddings": True,
            "precision": "float32",
            "batch_size": 1,
            "show_progress_bar": show_progress,
        }
        embed = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    elif "ollama" in model_name:
        #try tu pull the model from ollama using command ollama pull model_name
     
        embed = OllamaEmbeddings(model=model_name.split("/")[-1])

    elif "snowflake" in model_name or "Snowflake" in model_name:
        embed = CustomFastEmbedEmbeddings(model_name=model_name)

    else:
        encode_kwargs = {"batch_size": 16} #, "show_progress_bar": show_progress}
        embed = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    return embed

@lru_cache(maxsize=None)
def get_sparse_embedding_model(model_name, show_progress=False):
    """
    Returns a sparse embedding model based on the given model name.
    Args:
        model_name (str): The name of the model.
        show_progress (bool, optional): Whether to show progress or not. Defaults to False.
    Returns:
        embed (object): The sparse embedding model object.
    Raises:
        None
    Examples:
        >>> model = get_sparse_embedding_model("fastembed_sparse")
        >>> print(model)
        <FastEmbedSparse object at 0x7f9a2e3e4a90>
    """

    
    from langchain_community.embeddings import FastEmbedSparse

    embed = FastEmbedSparse(model_name=model_name)

    return embed


class CustomFastEmbedEmbeddings(BaseModel, Embeddings):
    """Qdrant FastEmbedding models.
    FastEmbed is a lightweight, fast, Python library built for embedding generation.
    See more documentation at:
    * https://github.com/qdrant/fastembed/
    * https://qdrant.github.io/fastembed/

    To use this class, you must install the `fastembed` Python package.

    `pip install fastembed`
    Example:
        from langchain_community.embeddings import FastEmbedEmbeddings
        fastembed = FastEmbedEmbeddings()
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    """Name of the FastEmbedding model to use
    Defaults to "BAAI/bge-small-en-v1.5"
    Find the list of supported models at
    https://qdrant.github.io/fastembed/examples/Supported_Models/
    """

    max_length: int = 512
    """The maximum number of tokens. Defaults to 512.
    Unknown behavior for values > 512.
    """

    cache_dir: Optional[str]
    """The path to the cache directory.
    Defaults to `local_cache` in the parent directory
    """

    threads: Optional[int]
    """The number of threads single onnxruntime session can use.
    Defaults to None
    """

    doc_embed_type: Literal["default", "passage"] = "default"
    """Type of embedding to use for documents
    The available options are: "default" and "passage"
    """

    _model: Any  # : :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = 'forbid'
        protected_namespaces = ()

    #@root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that FastEmbed has been installed."""
        model_name = values.get("model_name")
        max_length = values.get("max_length")
        cache_dir = values.get("cache_dir")
        threads = values.get("threads")

        try:
            # >= v0.2.0
            from fastembed import TextEmbedding

            values["_model"] = TextEmbedding(
                model_name=model_name,
                max_length=max_length,
                cache_dir=cache_dir,
                threads=threads,
                providers=["CUDAExecutionProvider"],
            )
        except ImportError as ie:
            try:
                # < v0.2.0
                from fastembed.embedding import FlagEmbedding

                values["_model"] = FlagEmbedding(
                    model_name=model_name,
                    max_length=max_length,
                    cache_dir=cache_dir,
                    threads=threads,
                )
            except ImportError:
                raise ImportError(
                    "Could not import 'fastembed' Python package. "
                    "Please install it with `pip install fastembed`."
                ) from ie
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents using FastEmbed.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings: List[np.ndarray]
        if self.doc_embed_type == "passage":
            embeddings = self._model.passage_embed(texts)
        else:
            embeddings = self._model.embed(texts)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        query_embeddings: np.ndarray = next(self._model.query_embed(text))
        return query_embeddings.tolist()


if __name__ == "__main__":
    model_name = "Snowflake/snowflake-arctic-embed-l"
    embedding_model = get_embedding_model(model_name)
    print(embedding_model)
