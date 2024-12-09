import logging
from functools import lru_cache
import torch

# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import (
     HuggingFaceBgeEmbeddings)
from langchain_huggingface import HuggingFaceEmbeddings

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
    model_kwargs = {"device": device,"trust_remote_code": True}

    if "bge" in model_name and "ollama" not in model_name:
        encode_kwargs = {
            "normalize_embeddings": True,
            "precision": "float32",
            "batch_size": 16,
            "show_progress_bar": show_progress,
        }
        embed = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    # elif "ollama" in model_name:
    #     #try tu pull the model from ollama using command ollama pull model_name
     
    #     embed = OllamaEmbeddings(model=model_name.split("/")[-1])

    # elif "snowflake" in model_name or "Snowflake" in model_name:
    #     embed = CustomFastEmbedEmbeddings(model_name=model_name)

    else:
        encode_kwargs = {"batch_size": 4} #, "show_progress_bar": show_progress}
        embed = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    return embed

@lru_cache(maxsize=None)
def get_sparse_embedding_model(model_name):
    """
    Returns a sparse embedding model based on the given model name.
    Args:
        model_name (str): The name of the model.
    Returns:
        embed (object): The sparse embedding model object.
    Raises:
        None
    Examples:
        >>> model = sparse_embedding_model("bge_model")
        >>> print(model)
        <HuggingFaceBgeEmbeddings object at 0x7f9a2e3e4a90>
    """
    from langchain_qdrant import FastEmbedSparse
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #model_kwargs = {"device": device}

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    
    return sparse_embeddings

if __name__ == "__main__":
    model_name = "Snowflake/snowflake-arctic-embed-l"
    embedding_model = get_embedding_model(model_name)
    print(embedding_model)
