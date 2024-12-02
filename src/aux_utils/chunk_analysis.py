import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from langchain_qdrant import Qdrant

from src.main_utils.embedding_utils import get_embedding_model
from src.main_utils.utils import get_k_random_chunks_qdrant

"""
chunk_analysis.py

This script performs analysis on text chunks from a dataset and visualizes the distribution of word and character counts 
within these chunks. It uses the Seaborn and Matplotlib libraries to create histograms that display the frequency 
distributions of the number of words and characters per chunk. The script also adds a new column to the DataFrame to 
indicate the source folder of each chunk.

Key functionalities:
- Add a new column to the DataFrame for the source folder of each chunk.
- Set a light background theme for the plots.
- Create and save a histogram of the distribution of the number of words per chunk.
- Create and save a histogram of the distribution of the number of characters per chunk.
"""

def load_config(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)


def load_embedding_model(config):
    return get_embedding_model(model_name=config["embedding_model"])


def load_qdrant_db(config, embedding_model):
    return Qdrant.from_existing_collection(
        path=config["persist_directory"],
        embedding=embedding_model,
        collection_name="qdrant_vectorstore",
    )


def get_documents_and_metadatas(k, config, qdrant_db):
    return get_k_random_chunks_qdrant(
        k=k, config=config, get_all=True, qdrant_db=qdrant_db
    )


def plot_graphs(documents, metadatas):
    # Create a DataFrame from documents and metadatas
    df = pd.DataFrame(
        {"text": documents, "source": [k["source"] for k in metadatas]}
    )

    # Add a new column for the size of the chunks in terms of number of words
    df["num_words"] = df["text"].apply(lambda x: len(x.split()))

    # Add a new column for the size of the chunks in terms of number of characters
    df["num_chars"] = df["text"].apply(len)

    # Add a new column for the source folder
    df["source_folder"] = df["source"].apply(lambda x: x.split("/")[1])

    # Set a light background theme for the plots
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 7))
    sns.histplot(df["num_words"], bins=50, kde=True, color="dodgerblue")
    plt.title("Distribution du nombre de mots par tronçons", fontsize=16)
    plt.xlabel("Nombre de Mots", fontsize=14)
    plt.ylabel("Fréquence", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("num_words_distribution.jpg", bbox_inches="tight")
    plt.close()

    # Plot the distribution of the sizes of the chunks in terms of number of characters
    plt.figure(figsize=(12, 7))
    sns.histplot(df["num_chars"], bins=50, kde=True, color="coral")
    plt.title("Distribution du nombre de caractères par tronçons", fontsize=16)
    plt.xlabel("Nombre de Caractères", fontsize=14)
    plt.ylabel("Fréquence", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("num_chars_distribution.jpg", bbox_inches="tight")
    plt.close()

    # Plot an histogram of the distribution of chunks per sources
    plt.figure(figsize=(12, 7))
    sns.countplot(y="source_folder", data=df, palette="viridis")
    plt.title("Distribution des tronçons par source", fontsize=16)
    plt.xlabel("Fréquence", fontsize=14)
    plt.ylabel("Dossier Source", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("chunks_per_source.jpg", bbox_inches="tight")
    plt.close()


def main():
    config = load_config("config/config.yaml")
    embedding_model = load_embedding_model(config)
    qdrant_db = load_qdrant_db(config, embedding_model)
    documents, metadatas = get_documents_and_metadatas(10, config, qdrant_db)
    plot_graphs(documents, metadatas)


if __name__ == "__main__":
    main()
