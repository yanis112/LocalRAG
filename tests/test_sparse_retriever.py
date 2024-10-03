from langchain_community.retrievers import (
    QdrantSparseVectorRetriever,
)
from langchain_core.documents import Document


from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from langchain_qdrant import Qdrant

import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer
#from src.custom_langchain_componants import initialize_sparse_vectorstore
#from src.embedding_model import get_embedding_model

def compute_sparse_vector(text, model_id="naver/splade-cocondenser-ensembledistil"):
        """
        Computes a vector from logits and attention mask using ReLU, log, and max operations.
        """
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        output = model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()

        # Convert vec to a list of indices and values for SparseVector
        indices = list(range(len(vec)))
        values = vec.tolist()
        
        return indices, values


if __name__ == "__main__":
    
    
    import yaml
    

    #load the config file
    with open("config/config.yaml") as f:
        default_config = yaml.safe_load(f)
        
    custom_persist=default_config["persist_directory"]+"/sparse_vector"
   
    
    #initialize_sparse_vectorstore(config=default_config,custom_persist=custom_persist)
    
    
    client = QdrantClient(path=custom_persist)
   
    
    retriever = QdrantSparseVectorRetriever(
        client=client,
        collection_name='sparse_vector',
        sparse_vector_name='text',
        sparse_encoder=compute_sparse_vector,
    )
    
    a=retriever.invoke("C'est quoi un réseau convolutif ?")
    
    print(a)
    
    
    exit()
    
        
    persist_dir = default_config["persist_directory"]

    # Qdrant client setup
    client = QdrantClient(path=persist_dir)
    
    try:
        client.create_collection(
            collection_name="sparse_vector_v6",
            init_from=models.InitFrom(collection="qdrant_vectorstore"),
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            sparse_vectors_config={
                'text': models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,
                    )
                )
            },
        )
        print("Collection already exists !")
    except Exception as e:
        print(e)
        print("Collection  does not already exists !")
        client.get_collection(collection_name="sparse_vector_v6")


    print("e")

    # Create a retriever with the new encoder
    retriever = QdrantSparseVectorRetriever(
        client=client,
        collection_name='sparse_vector_v6',
        sparse_vector_name='text',
        sparse_encoder=compute_sparse_vector,
    )
        # Ajoutez des documents et vérifiez la réponse
    documents = [
        Document(
            metadata={"title": "Document 1"},
            page_content="J'aime bien les vaches."
        ),
        Document(
            metadata={"title": "Document 2"},
            page_content="comment coder un réseau de neurone"
        )
    ]
    #retriever.add_documents(documents)
    
    answer=retriever.invoke("C'est quoi un réseau convolutif ?")
    print(answer)