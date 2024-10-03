import json

import matplotlib.pyplot as plt
import umap.umap_ as umap
from FlagEmbedding import BGEM3FlagModel
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import numpy as np


def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    queries = [v["query"] for v in data.values()]
    labels = [v["label"] for v in data.values()]
    return queries, labels


def create_embeddings(queries):
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    embeddings = model.encode(queries, batch_size=12, max_length=8192)['dense_vecs']
    return embeddings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import umap

def plot_embeddings_by_label(embeddings, labels):
    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels_encoded,
        cmap="Spectral",
        s=5,
    )
    plt.gca().set_aspect("equal", "datalim")

    # Create a legend
    colors = [scatter.cmap(scatter.norm(i)) for i in range(len(le.classes_))]
    patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in colors]
    plt.legend(patches, le.classes_, title="Labels")

    plt.title("UMAP projection of the sentence embeddings by label", fontsize=24)
    plt.savefig("umap_label.png")
    plt.show()
    plt.close()

def plot_embeddings_by_cluster(embeddings):
    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(embeddings)

    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=kmeans.labels_,
        cmap="Spectral",
        s=5,
    )
    plt.gca().set_aspect("equal", "datalim")

    # Create a legend
    colors = [scatter.cmap(scatter.norm(i)) for i in range(2)]
    patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in colors]
    plt.legend(patches, ['Cluster 1', 'Cluster 2'], title="Clusters")

    plt.title("UMAP projection of the sentence embeddings by cluster", fontsize=24)
    plt.savefig("umap_cluster.png")
    plt.show()
    #close the plot
    plt.close()

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import umap
import numpy as np

import matplotlib.pyplot as plt

from collections import Counter
import random

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import umap
import random

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import umap
import random

def find_nearest_cluster(query, embeddings, labels, n_clusters=2):
    # Convert labels to a NumPy array
    labels = np.array(labels)

    # Create a KMeans model
    kmeans = KMeans(n_clusters=n_clusters)
    # Fit the model to the embeddings
    kmeans.fit(embeddings)

    # Create an embedding for the query
    query_embedding = create_embeddings([query]) # This function is not defined in the provided code

    # Calculate the distance to each cluster center
    distances = np.linalg.norm(kmeans.cluster_centers_ - query_embedding, axis=1)

    # Determine the majority label for each cluster
    cluster_labels = []
    for i in range(n_clusters):
        cluster_points = labels[kmeans.labels_ == i]
        majority_label = Counter(cluster_points).most_common(1)[0][0]
        cluster_labels.append(majority_label)

    # Find the indices of the 'lexical' and 'semantic' clusters
    lexical_index = cluster_labels.index('lexical')
    semantic_index = cluster_labels.index('semantic')

    # Get the cluster centers
    lexical_center = kmeans.cluster_centers_[lexical_index]
    semantic_center = kmeans.cluster_centers_[semantic_index]

    # Project the query point onto the line connecting the two cluster centers
    line_vector = semantic_center - lexical_center
    query_vector = query_embedding - lexical_center
    projection_length = np.dot(query_vector, line_vector) / np.linalg.norm(line_vector)

    # Normalize the projection length to get a score between 0 and 1
    total_length = np.linalg.norm(semantic_center - lexical_center)
    score = projection_length / total_length

    # Find the index of the nearest cluster
    nearest_index = distances.argmin()
    # Find the label of the nearest cluster
    nearest_label = cluster_labels[nearest_index]

    # Reduce the dimensionality of the embeddings with UMAP for visualization
    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings)
    query_embedding_2d = reducer.transform(query_embedding)

    # Plot the clusters and the query point
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(query_embedding_2d[0, 0], query_embedding_2d[0, 1], c='red')
    plt.title('Clusters and query point')
    plt.savefig("clusters_query_"+str(random.randint(0,1000))+".png")
    plt.show()
    plt.close()

    return nearest_index, nearest_label, score

    
if __name__ == "__main__":
    queries, labels = load_data("./src/datasets/query_router_finetuning_dataset.json")
    embeddings = create_embeddings(queries)
    plot_embeddings_by_label(embeddings, labels)
    plot_embeddings_by_cluster(embeddings)
    

    query1="Quels sont les caractéristiques de la plateforme Digazu ?"
    query2="Comment se connecter à une machine distante ?"
    query3="Qui est Alice ?"
    query4=" Qui est Guillaume Lamazou ? "
    list_query=[query1,query2,query3,query4]
    for query in list_query:
        cluster, nearest_label, score = find_nearest_cluster(query, embeddings, labels)
        print(f"Query: {query}, Cluster: {cluster}, Nearest Label: {nearest_label}, Score: {score}")
 
