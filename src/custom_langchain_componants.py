import logging
import operator
import os
import warnings
from collections import defaultdict
from collections.abc import Hashable
from functools import lru_cache
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
import torch
from dotenv import load_dotenv
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    BaseCrossEncoder,
)
from langchain_community.retrievers import (
    QdrantSparseVectorRetriever,
)
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    Callbacks,
)
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.load.dump import dumpd
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import Field
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from langchain_qdrant import Qdrant
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.embedding_model import get_embedding_model

# custom imports
from src.utils import log_execution_time

# Load the environment variables (API keys, etc...)
load_dotenv()


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # solve too many warnings

# defining the Sparse Encoder
@lru_cache(maxsize=None)
def load_sparse_encoder(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    return model, tokenizer


def default_preprocessing_func(text: str) -> List[str]:
    # Default implementation of the preprocessing function
    return text.lower().split()



class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return lines


# Default prompt
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is 
    to generate 3 different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {question}""",
)


def _unique_documents(documents: Sequence[Document]) -> List[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


def token_calculation_prompt(query: str) -> str:
    """
    Return the appropriate token calculation prompt for the query or document.
    """
    coefficient = 1 / 0.45
    num_tokens = len(query.split()) * coefficient
    return num_tokens

class TopKCompressor(BaseDocumentCompressor):
    """Document compressor that returns the top k documents.
    method: compress_documents takes a list of langchain Documents objects and returns al list
    of the top k documents.
    """

    k: int = Field(...)
    intelligent_compression: bool = False
    token_target: int = 4000

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: Optional[str] = None,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Return the top k documents."""
        if not self.intelligent_compression:
            return documents[: self.k]

        selected_documents = []
        total_tokens = 0
        for doc in documents:
            tokens = token_calculation_prompt(doc.page_content)
            if abs(total_tokens + tokens - self.token_target) < abs(
                total_tokens - self.token_target
            ):
                total_tokens += tokens
                selected_documents.append(doc)
            else:
                break

        return selected_documents

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: Optional[str] = None,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Return the top k documents asynchronously."""
        if not self.intelligent_compression:
            return documents[: self.k]

        selected_documents = []
        total_tokens = 0
        for doc in documents:
            tokens = await token_calculation_prompt(doc.page_content)
            if abs(total_tokens + tokens - self.token_target) < abs(
                total_tokens - self.token_target
            ):
                total_tokens += tokens
                selected_documents.append(doc)
            else:
                break

        return selected_documents


class CustomCrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder for reranking."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    top_n: int = 3
    """Number of documents to return."""
    intelligent_compression: bool = False
    """Whether to use intelligent compression based on token count."""
    token_target: int = 4000
    """Target token count for intelligent compression."""
    use_autocut: bool = False
    """Whether to use autocut for document selection."""
    autocut_beta: float = 2.5
    """Beta value for determining significant drop in autocut."""

    class Config:
        """Configuration for this pydantic object."""

        extra = 'forbid'   #Extra.forbid
        arbitrary_types_allowed = True
        

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of Langchain document objects to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """

        scores = self.model.score(
            [(query, doc.page_content) for doc in documents]
        )
        docs_with_scores = list(zip(documents, scores))
        sorted_docs_with_scores = sorted(
            docs_with_scores, key=operator.itemgetter(1), reverse=True
        )
        
        # Plotting the scores
        import matplotlib.pyplot as plt
        scores_sorted = [score for _, score in sorted_docs_with_scores]
        
        # Calculate the autocut index
        autocut_index = self.autocut_v2(scores_sorted, min_docs=3)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(scores_sorted) + 1), scores_sorted, marker='o')
        if autocut_index is not None:
            plt.plot(autocut_index + 1, scores_sorted[autocut_index], 'ro')  # Mark the drop point in red
        plt.xlabel('Document Number')
        plt.ylabel('Score')
        plt.title('Document Scores from Highest to Lowest')
        plt.grid(True)
        plt.savefig('document_scores.png')
        plt.close()

        if self.use_autocut and autocut_index is not None:
            return [doc for doc, _ in sorted_docs_with_scores[:autocut_index]]

        if not self.intelligent_compression:
            return [doc for doc, _ in sorted_docs_with_scores[: self.top_n]]

        selected_documents = []
        total_tokens = 0
        for doc, _ in sorted_docs_with_scores:
            tokens = token_calculation_prompt(doc.page_content)
            if abs(total_tokens + tokens - self.token_target) < abs(
                total_tokens - self.token_target
            ):
                total_tokens += tokens
                selected_documents.append(doc)
            else:
                break

        return selected_documents

    def autocut(self, scores_sorted: list, beta: float) -> Optional[int]:
        """
        Calculate the index where the drop is greater than beta times the standard deviation of the drops.

        Args:
            scores_sorted: List of scores sorted in descending order.
            beta: Multiplier for the standard deviation to determine significant drop.

        Returns:
            The index where the significant drop occurs, or None if no such drop is found.
        """
        baisses = [scores_sorted[i] - scores_sorted[i + 1] for i in range(len(scores_sorted) - 1)]
        std_baisses = np.std(baisses)

        for i, baisse in enumerate(baisses):
            if baisse > beta * std_baisses:
                return i + 1  # Return the index where the drop occurs

        return None
    
    def autocut_v2(self, scores_sorted: list, min_docs: int = 1) -> int:
        """
        Calculate the index where the largest drop occurs, ensuring a minimum number of documents retrieved.
    
        Args:
            scores_sorted: List of scores sorted in descending order.
            min_docs: Minimum number of documents to retrieve before auto-cutting.
    
        Returns:
            The index before the largest drop occurs, or min_docs if the calculated index is smaller than min_docs.
        """
        if len(scores_sorted) < min_docs:
            return len(scores_sorted)  # Return the total number of documents if less than min_docs
    
        baisses = [scores_sorted[i] - scores_sorted[i + 1] for i in range(len(scores_sorted) - 1)]
        if not baisses:
            return len(scores_sorted)  # Return the total number of documents if no drops are found
    
        max_baisse_index = np.argmax(baisses)
        return max(max_baisse_index + 1, min_docs)  # Ensure the index is at least min_docs

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """Yield unique elements of an iterable based on a key function.

    Args:
        iterable: The iterable to filter.
        key: A function that returns a hashable key for each element.

    Yields:
        Unique elements of the iterable based on the key function.
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


class CustomEnsembleRetriever(BaseRetriever):
    """Retriever that ensembles the multiple retrievers.

    It uses a rank fusion.

    Args:
        retrievers: A list of retrievers to ensemble.
        weights: A list of weights corresponding to the retrievers. Defaults to equal
            weighting for all retrievers.
        c: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
            Default is 60.
    """

    retrievers: List[RetrieverLike]
    weights: List[float]
    c: int = 60

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        """List configurable fields for this runnable."""
        return get_unique_config_specs(
            spec
            for retriever in self.retrievers
            for spec in retriever.config_specs
        )

    def set_weights(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("weights"):
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1 / n_retrievers] * n_retrievers
        return values

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        from langchain_core.callbacks import CallbackManager

        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            **kwargs,
        )
        try:
            result = self.rank_fusion(
                input, run_manager=run_manager, config=config
            )
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    def relative_rank_fusion(
        self, doc_lists: List[List[Document]]
    ) -> List[Document]:
        """
        Perform Relative Rank Fusion on multiple rank lists.
        This method normalizes the scores from vector and keyword searches,
        scaling them between 0 (lowest) to 1 (highest) before combining.

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their normalized scores
                    in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # Associate each doc's content with its score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively
        score: Dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                score[doc.page_content] += weight * doc.score

        # Normalize the scores between 0 and 1
        scaler = MinMaxScaler()
        scores = np.array(list(score.values())).reshape(-1, 1)
        normalized_scores = scaler.fit_transform(scores)

        for doc_content, norm_score in zip(score, normalized_scores):
            score[doc_content] = norm_score[0]

        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(all_docs, lambda doc: doc.page_content),
            reverse=True,
            key=lambda doc: score[doc.page_content],
        )
        return sorted_docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get fused result of the retrievers.
        fused_documents = self.relative_rank_fusion(query, run_manager)

        return fused_documents

    def rank_fusion(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """
        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            retriever.invoke(
                query,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(tag=f"retriever_{i+1}"),
                ),
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast(str, doc))
                if isinstance(doc, str)
                else doc
                for doc in retriever_docs[i]
            ]

        # apply rank fusion
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)

        return fused_documents

    def weighted_reciprocal_rank(
        self, doc_lists: List[List[Document]]
    ) -> List[Document]:
        """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively
        rrf_score: Dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[doc.page_content] += weight / (rank + self.c)

        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(all_docs, lambda doc: doc.page_content),
            reverse=True,
            key=lambda doc: rrf_score[doc.page_content],
        )
        return sorted_docs


def extract_and_map_sparse_vector(vector, tokenizer):
    """
    Extracts non-zero elements from a given vector and maps these elements to their human-readable tokens using a tokenizer. The function creates and returns a sorted dictionary where keys are the tokens corresponding to non-zero elements in the vector, and values are the weights of these elements, sorted in descending order of weights.

    This function is useful in NLP tasks where you need to understand the significance of different tokens based on a model's output vector. It first identifies non-zero values in the vector, maps them to tokens, and sorts them by weight for better interpretability.

    Args:
    vector (torch.Tensor): A PyTorch tensor from which to extract non-zero elements.
    tokenizer: The tokenizer used for tokenization in the model, providing the mapping from tokens to indices.

    Returns:
    dict: A sorted dictionary mapping human-readable tokens to their corresponding non-zero weights.
    """

    # Extract indices and values of non-zero elements in the vector
    cols = vector.nonzero().squeeze().cpu().tolist()
    weights = vector[cols].cpu().tolist()

    # Map indices to tokens and create a dictionary
    idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}
    token_weight_dict = {
        idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
    }

    # Sort the dictionary by weights in descending order
    sorted_token_weight_dict = {
        k: v
        for k, v in sorted(
            token_weight_dict.items(), key=lambda item: item[1], reverse=True
        )
    }

    return sorted_token_weight_dict


def compute_sparse_vector(
    text,
    model_id="naver/splade-cocondenser-ensembledistil",
):
    """
    Computes a sparse vector from logits and attention mask using ReLU, log, and max operations.

    Args:
        text (str): The input text to compute the sparse vector.
        model_id (str, optional): The ID of the pre-trained model to use. Defaults to "naver/splade-cocondenser-ensembledistil".
        tokenizer (Tokenizer, optional): The tokenizer to use for tokenizing the text. Defaults to tokenizer.
        model (Model, optional): The pre-trained model to use for computing the logits and attention mask. Defaults to model.

    Returns:
        tuple: A tuple containing the indices and values of the sparse vector.
    """
    
    #load the model and tokenizer (cached)
    model, tokenizer = load_sparse_encoder(model_id)
    device="cuda" if torch.cuda.is_available() else "cpu"

    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()

    # Extract non-zero indices and values for SparseVector
    non_zero_indices = vec.nonzero(as_tuple=False).squeeze().cpu().tolist()
    non_zero_values = vec[non_zero_indices].cpu().tolist()

    return non_zero_indices, non_zero_values


@log_execution_time
def initialize_sparse_vectorstore(
    config,
    use_embedding_quatization=False,
):
    """
    Initializes a sparse vector store using the provided configuration using the chunks from the already existing dense vector store, only if the sparse vector store does not already exist.

    Args:
        config (dict): The configuration dictionary.
        custom_persist (str): The path to the custom persistence directory.
        embedding_size (int, optional): The size of the embedding vectors. Defaults to 1024.
        model_id (str, optional): The ID of the embedding model. Defaults to "naver/splade-cocondenser-ensembledistil".

    Returns:
        None
    """
    from src.utils import get_all_docs_qdrant
    
    embedding_model = get_embedding_model(model_name=config["embedding_model"])

    # separate clients to avoid conflicts !
    raw_database = Qdrant.from_existing_collection(
        path=config["persist_directory"],
        embedding=embedding_model,
        collection_name="qdrant_vectorstore",
    )

    #get all the documents from the database instead of reproccessing them
    documents = get_all_docs_qdrant(raw_database=raw_database)

    doc_copy = documents.copy()
    del raw_database  # USEFULL OR NOT ??
    
    #We create the custom persist for the sparse vector based on the one from the dense
    sparse_persist = config['persist_directory'] + "_sparse_vector"

    # Qdrant client setup
    client = QdrantClient(path=sparse_persist)

    # Check if the sparse vector collection already exists
    if client.collection_exists(collection_name="sparse_vector"):
        print("Sparse vector collection already exists. Skipping initialization !")
        return

    #initialize the sparse vector collection
    client.create_collection(
        collection_name="sparse_vector",
        vectors_config=models.VectorParams(
            size=config["sparse_embedding_size"],
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True,
                )
            )
        },
    )
    
    from functools import partial
    compute_sparse_vector_with_model = partial(compute_sparse_vector, model_id=config["sparse_embedding_model"])

    # Create a retriever with the new encoder
    retriever = QdrantSparseVectorRetriever(
        client=client,
        collection_name="sparse_vector",
        sparse_vector_name="text",
        sparse_encoder=compute_sparse_vector_with_model
        #compute_sparse_vector(model_id=config["sparse_embedding_model"]),
    )

    retriever.add_documents(documents=doc_copy)