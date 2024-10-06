import hdbscan
import pandas as pd
import plotly.express as px
import umap
import yaml
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from langchain_qdrant import Qdrant
import textwrap


from src.utils import get_k_random_chunks,get_k_random_chunks_qdrant
from src.embedding_model import get_embedding_model


"""
clustering_by_topic.py

This script is designed to visualize clustering results of a dataset in a 2D space. It uses Plotly to create scatter plots
that display the data points colored by their assigned clusters or topics. The script defines a color map to ensure that 
each unique topic or cluster is represented by a distinct color. The visualizations include hover data to provide additional 
context about each data point, such as its topic, source, and chunk.

Key functionalities:
- Define a color map with enough distinct colors for all unique topics.
- Create a scatter plot to visualize data points colored by clusters.
- Create a scatter plot to visualize data points colored by topics.
"""


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_embeddings(chunks, model_name="all-mpnet-base-v2"):
    """Get embeddings for the chunks using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings


def reduce_embeddings(embeddings):
    """Reduce the dimensionality of embeddings using UMAP."""
    umap_model = umap.UMAP(
        n_neighbors=50, n_components=2, metric="cosine"
    ).fit_transform(embeddings)
    return umap_model


def cluster_embeddings(reduced_embeddings, min_cluster_size=15):
    """Cluster the reduced embeddings using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean"
    )
    clusters = clusterer.fit_predict(reduced_embeddings)
    return clusters



def plot_embeddings(reduced_embeddings, clusters, topics, topic_names, sources, chunks):
    """Plot the reduced embeddings in 2D with cluster colors and topic names."""
    # Wrap each chunk into lines of 15 words
    chunks_wrapped = ['<br>'.join(textwrap.wrap(chunk, width=80)) for chunk in chunks]

    topic_name_list = [
        topic_names[topic] if topic in topic_names else "Noise"
        for topic in topics
    ]
    df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "cluster": pd.Categorical(clusters),  # Convert to categorical
            "topic": topic_name_list,  # Convert to categorical
            "source": sources,  # Add 'source' to the DataFrame
            "chunk": chunks_wrapped,  # Add 'chunks' to the DataFrame
        }
    )

    # Define a color map (you can change the colors as needed)
    color_map_cluster = {category: color for category, color in zip(df['cluster'].unique(), px.colors.qualitative.G10)}
    
    # Define a color map with enough distinct colors
    unique_topics = df['topic'].unique()
    #color_palette = px.colors.qualitative.Alphabet[:len(unique_topics)]
    color_palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Safe + px.colors.qualitative.Vivid
    color_map_topic = {category: color for category, color in zip(unique_topics, color_palette)}
    color_map_cluster = {category: color for category, color in zip(unique_topics, color_palette)}


    # Plot with clusters
    # fig1 = px.scatter(
    #     df,
    #     x="x",
    #     y="y",
    #     color="cluster",
    #     hover_data=["topic", "source", "chunk"],
    #     title="2D Embeddings with Clusters",
    #     color_discrete_map=color_map_cluster  # Use the color map
    # )  # Include 'source' and 'chunk' in hover_data
    # fig1.update_traces(
    #     marker=dict(size=10, opacity=0.8), selector=dict(mode="markers")
    # )
    # fig1.show()

    # Plot with topics
    fig2 = px.scatter(
        df,
        x="x",
        y="y",
        color="topic",
        hover_data=["cluster", "source", "chunk"],
        title="2D Embeddings with Topics",
        color_discrete_map=color_map_topic  # Use the color map
    )  # Include 'source' and 'chunk' in hover_data
    fig2.update_traces(
        marker=dict(size=10, opacity=0.8), selector=dict(mode="markers")
    )
    fig2.show()


def main():
    # Load the configuration
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # Get the chunks and metadata
    k = 1
    embedding_model = get_embedding_model(model_name=config["embedding_model"])
    
    qdrant_db= Qdrant.from_existing_collection(
            path=config["persist_directory"],
            embedding=embedding_model,
            collection_name="qdrant_vectorstore",
        )
        
    # list_chunks, list_metadata = get_k_random_chunks(
    #     k,
    #     config,
    #     get_all=True,
    #     clone_persist=None,
    #     clone_embedding_model=None,
    #     return_embeddings=False,
    #     special_embedding=None,
    # )
    
    # get k random chunks using Qdrant
    list_chunks, list_metadata = get_k_random_chunks_qdrant(
        k=k, config=config, get_all=True, qdrant_db=qdrant_db
    )
    
    #remove the chunks (and associated metadata) that are less than 40 words
    list_chunks = [chunk for chunk in list_chunks if len(chunk.split()) >= 40]
    list_metadata = [metadata for metadata, chunk in zip(list_metadata, list_chunks) if len(chunk.split()) >= 40]
    
    print("Chunks Obtained")
    # Get embeddings
    list_embeddings = get_embeddings(list_chunks)

    # Initialize and use BERTopic with the obtained embeddings
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(list_chunks, list_embeddings)
    dataframe_topic = topic_model.get_topic_info()

    # Create a dictionary mapping topic IDs to topic names
    topic_names = dataframe_topic["Name"].to_dict()

    # Print unique topic names
    print("UNIQUE TOPICS", dataframe_topic["Name"].unique())

    # Reduce embeddings to 2D
    reduced_embeddings = reduce_embeddings(list_embeddings)

    # Cluster reduced embeddings
    clusters = cluster_embeddings(reduced_embeddings)

    # Extract 'source' from metadata
    sources = [metadata["source"] for metadata in list_metadata]

    # Identify indices of 'Noise' class
    noise_indices = [i for i, cluster in enumerate(clusters) if cluster == -1]

    # print how much noisy documents we have
    print("Number of noisy documents:", len(noise_indices))

    # Print 10 document chunks from 'Noise' class
    print("10 document chunks from 'Noise' class:")
    for i in noise_indices[:10]:
        print("##################################################")
        print("DOCUMENT CHUNK", i)
        print(list_chunks[i])

    # Plot the embeddings
    plot_embeddings(
    reduced_embeddings, clusters, topics, topic_names, sources, list_chunks )

if __name__ == "__main__":
    main()
