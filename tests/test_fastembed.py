from fastembed import TextEmbedding
import pandas as pd

# Example list of documents
documents: list[str] = [
    "This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
    "fastembed is supported by and maintained by Qdrant.",
]

supported_models = (
    pd.DataFrame(TextEmbedding.list_supported_models())
    .sort_values("size_in_GB")
    .drop(columns=["sources", "model_file", "additional_files"])
    .reset_index(drop=True)
)
print(supported_models)

# exit()
# # This will trigger the model download and initialization
# embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v3")
# print("The model BAAI/bge-small-en-v1.5 is ready to use.")

# embeddings_generator = embedding_model.embed(documents)  # reminder this is a generator
# embeddings_list = list(embedding_model.embed(documents))
#   # you can also convert the generator to a list, and that to a numpy array
# len(embeddings_list[0]) # Vector of 384 dimensions