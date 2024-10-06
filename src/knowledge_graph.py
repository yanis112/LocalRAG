import json
import os
import re
import shutil
from functools import lru_cache
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ollama
import yaml
from gliner import GLiNER
from grandcypher import GrandCypher
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_qdrant import Qdrant
from networkx.drawing.nx_pydot import graphviz_layout
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz
from tqdm import tqdm


from src.embedding_model import get_embedding_model
from src.generation_utils import LLM_answer_v3, entity_description


class KnowledgeTriple(BaseModel):
    list_relations: List[tuple] = Field(
        description="List of relations as knowledge triples of strings: ('subject', 'predicate', 'object')"
    )


class KnowledgeGraph:
    """
    Represents a knowledge graph that stores and manages knowledge triples.

    Attributes:
        triplets (List[tuple]): A list of knowledge triples in the form of (subject, predicate, object).
        graph (nx.DiGraph): A directed graph representation of the knowledge graph.

    Methods:
        __init__(self, triplets: Optional[List[tuple]] = None): Initializes a KnowledgeGraph object.
        add_triplet(self, triplet: tuple): Adds a knowledge triplet to the graph.
        generate_from_text(self, text: str): Generates knowledge triples from the given text.
        save_figure(self): Saves a visual representation of the knowledge graph as an image.
        get_triplets(self): Returns the list of knowledge triples.
        graph_to_dict(self): Converts the knowledge graph to a dictionary representation.

    """

    def __init__(
        self,
        triplets: Optional[List[tuple]] = None,
        model_name="gemma2",
        llm_provider="ollama",
    ):
        """
        Initializes a KnowledgeGraph object.

        Args:
            triplets (Optional[List[tuple]]): A list of knowledge triples in the form of (subject, predicate, object).
                Defaults to None.

        """

        config = yaml.safe_load(open("config/config.yaml"))
        self.triplets = triplets if triplets else []
        self.graph = nx.DiGraph()
        for triplet in self.triplets:
            self.add_triplet(triplet)
        self.model_name = config["model_name"]
        self.llm_provider = config["llm_provider"]

    def add_triplet(self, triplet: tuple):
        """
        Adds a knowledge triplet to the graph.

        Args:
            triplet (tuple): A knowledge triplet in the form of (subject, predicate, object).

        """

        def clean_string(string):
            # Remove all characters that are not letters, digits, a single space, or accented characters
            cleaned_string = re.sub(r"[^a-zA-Z0-9\s\u00C0-\u00FF]", "", string)
            # Replace multiple spaces with a single space
            cleaned_string = re.sub(r"\s+", " ", cleaned_string)
            return str(cleaned_string.strip())

        try:
            print
            if len(triplet) == 3:
                subject, predicate, obj = triplet
                clean_subject = clean_string(subject)
                clean_predicate = clean_string(predicate)
                clean_obj = clean_string(obj)

                # print("CURRENT GRAPH NODES:",self.graph.nodes())

                # Check if the subject or object already exists in the graph
                for node in self.graph.nodes():
                    if fuzz.token_set_ratio(clean_subject, node) > 60:
                        # Merge the new subject node with the existing one
                        self.graph = nx.contracted_nodes(
                            self.graph, clean_subject, node, self_loops=False
                        )
                        clean_subject = node

                    if fuzz.token_set_ratio(clean_obj, node) > 60:
                        # Merge the new object node with the existing one
                        self.graph = nx.contracted_nodes(
                            self.graph, clean_obj, node, self_loops=False
                        )
                        clean_obj = node

                # print("Adding edge:", clean_subject, clean_obj, clean_predicate)
                self.graph.add_edge(
                    clean_subject, clean_obj, label=clean_predicate
                )

            else:
                print(
                    "Error in triplet:",
                    triplet,
                    "there are more than 3 elements. Skipping...",
                )

        except Exception as ex:
            print("Error in triplet:", triplet, "Skipping...")
            print("Error type:", ex)

    def complete_from_text(self, text: str):
        """
        Generates knowledge triples from the given text.

        Args:
            text (str): The input text from which knowledge triples are extracted.

        """
        document = text
        prompt = f"""
        You are an AI assistant expert in relation extraction whose goal is to extract all the relational triples from a text.
        A knowledge triple is a clause that contains a subject, a predicate,
        and an object. The subject is the entity being described,
        the predicate is the property of the subject that is being
        described, and the object is the value of the property. If you can build relationships between the existing knowledge graph and the new text, do so, if you spot a missing relationship in the current knowledge graph, add it to the list.

        EXAMPLE
        It's a state in the US. It's also the number 1 producer of gold in the US.

        Output: [('Nevada', 'is a', 'state'),('Nevada', 'is in', 'US'),('Nevada', 'is the number 1 producer of', 'gold')]
        END OF EXAMPLE

        EXAMPLE
        Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.

        Output: [('Descartes', 'likes to drive', 'antique scooters'),('Descartes', 'plays', 'mandolin')]
        END OF EXAMPLE

        Here is the current knowledge graph: 
        
        {str(self.graph_to_dict())}
        
        And here is the text to process:
        
        {document}
        Output:
        """

        result = LLM_answer_v3(
            prompt,
            llm_provider=self.llm_provider,
            model_name=self.model_name,
            temperature=0,
            stream=False,
        )
        # print("Result:", result)

        triplets = re.findall(r"\(([^)]+)\)", result)

        # Convertit chaque tuple en une liste de ses éléments
        self.triplets = [
            tuple(
                [
                    str(k).replace("'", "").replace('"', "")
                    for k in triplet.split(",")
                ]
            )
            for triplet in triplets
        ]

        # print("NEW TRIPLETS:",self.triplets)

        for triplet in self.triplets:
            self.add_triplet(triplet)

    def contextualize_query(self, query: str) -> str:
        """
        Refines the given query based on the knowledge graph.

        Args:
            query (str): The user's query to be refined.

        Returns:
            str: The refined query.
        """
        # Convert the graph to a dictionary representation
        graph_dict = self.graph_to_dict()
        print("GRAPH DICT:", graph_dict)

        # Construct the refining prompt
        refining_prompt = f"""
        Based on the following question asked by a user: <query> {query} </query>, and on the following knowledge base (if any elements are relevant): <axioms> {str(graph_dict)} </axioms>, reformulate the question posed by the user to make it more precise, more relevant, and add any necessary context for its understanding.
        """

        # Assuming LLM_answer is a function or method that sends the prompt to a language model and returns a response
        class LLM_answer_response(BaseModel):
            refined_query: str

        # This part of the code is pseudo-code as the implementation details of LLM_answer are not provided
        llm_answer = LLM_answer_v3(
            refining_prompt,
            json_formatting=True,
            pydantic_object=LLM_answer_response,  # Assuming this is defined elsewhere
            llm_provider=self.llm_provider,
            model_name=self.model_name,
            stream=False,
            temperature=0,  # 0 here because we want the most precise answer and no creativity
        )

        # Extract the refined query from the language model's response
        refined_query = llm_answer["refined_query"]

        # Return the refined query
        return refined_query

    def save_figure(self, name="knowledge_graph.png"):
        """
        Saves a visual representation of the knowledge graph as an image.

        """
        # Convertir le graphe NetworkX en un graphe AGraph
        # G = to_agraph(self.graph)
        G = self.graph
        # print("GRAPH:",G)

        # Print all nodes and edges
        # print("Nodes:", G.nodes())
        # print("Edges:", G.edges())

        # clear all the data in the nodes and edges
        for node in G.nodes():
            G.nodes[node].clear()
        for edge in G.edges():
            G.edges[edge].clear()

        # remove edges with a '' element
        for k in list(G.edges()):
            if "" in k:
                G.remove_edge(k[0], k[1])
        # si un noeud n'a pas de relation avec un autre noeud, on le supprime
        for node in list(G.nodes()):
            if G.degree(node) == 0:
                G.remove_node(node)

        # print("Node data:", G.nodes(data=True))
        # print("Edge data:", G.edges(data=True))

        pos = graphviz_layout(G, prog="neato")

        plt.figure(figsize=(10, 10))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=800,
            font_size=10,
        )
        plt.savefig(name, dpi=300)

    def get_triplets(self):
        """
        Returns the list of knowledge triples.

        Returns:
            List[tuple]: A list of knowledge triples in the form of (subject, predicate, object).

        """
        return self.triplets

    def graph_to_dict(self):
        """
        Converts the knowledge graph to a dictionary representation.

        Returns:
            dict: A dictionary representation of the knowledge graph.

        """
        if self.triplets != []:
            G = {}
            for triplet in self.triplets:
                try:
                    subject, predicate, obj = triplet
                    if subject in G:
                        if predicate in G[subject]:
                            G[subject][predicate].append(obj)
                        else:
                            G[subject][predicate] = [obj]
                    else:
                        G[subject] = {predicate: [obj]}
                except:
                    print("Error in triplet:", triplet, "Skipping...")
        else:
            G = {"Empty graph": "Knowledge graph is empty for now"}

        return G

    def interactive_graph(self):
        import networkx as nx
        import plotly.graph_objs as go

        # Calculer la position des nœuds avec le layout spring en 3D
        pos = nx.spring_layout(self.graph, dim=3)

        # Initialiser les listes pour les coordonnées des arêtes et des nœuds
        edge_x, edge_y, edge_z = [], [], []
        node_x, node_y, node_z = [], [], []
        node_info = []  # Pour stocker le nom de l'entité associée à chaque nœud

        # Boucle pour ajouter les coordonnées des arêtes
        for edge in self.graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # Ajoute les coordonnées x de l'arête
            edge_y.extend([y0, y1, None])  # Ajoute les coordonnées y de l'arête
            edge_z.extend([z0, z1, None])  # Ajoute les coordonnées z de l'arête

        # Créer une trace pour les arêtes en 3D
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Boucle pour ajouter les coordonnées et les infos des nœuds
        for node in self.graph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_info.append(
                f"{node}"
            )  # Ajoute le nom de l'entité comme info à afficher au survol

        # Créer une trace pour les nœuds en 3D
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(thickness=15, title="Node Connections"),
                line_width=2,
            ),
        )

        # Ajuster la taille des nœuds basée sur le nombre de connexions, avec une taille minimale
        node_adjacencies = []
        for node, adjacencies in enumerate(self.graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        node_trace.marker.size = [
            10 + len(self.graph[node]) * 2 for node in self.graph.nodes()
        ]  # Ajuster la taille ici
        node_trace.marker.color = node_adjacencies
        node_trace.text = (
            node_info  # Le texte de survol contenant le nom de l'entité
        )

        # Créer la figure avec les traces des nœuds et des arêtes en 3D
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<br>Interactive Knowledge Graph in 3D",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                scene=dict(
                    xaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False
                    ),
                    yaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False
                    ),
                    zaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False
                    ),
                ),
            ),
        )

        # Afficher la figure
        fig.show()


def format_triplex_input(text, entity_types, predicates):
    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
      
        **Entity Types:**
        {entity_types}
        
        **Predicates:**
        {predicates}
        
        **Text:**
        {text}
        """
    return input_format.format(
        entity_types=json.dumps({"entity_types": entity_types}),
        predicates=json.dumps({"predicates": predicates}),
        text=text,
    )



# def parse_output_to_triples(output):
#     """
#     Parse the output string and extract triples from it.

#     Args:
#         output (str): The output string to parse.

#     Returns:
#         list: A list of triples extracted from the output string.

#     Raises:
#         None

#     The function takes an output string and extracts triples from it. It assumes that the output string
#     contains a JSON object with a key "entities_and_triples" which holds a list of strings representing
#     entities and relationships. The function iterates over the list and identifies entities and relationships
#     based on certain patterns. It then constructs triples using the identified entities and relationships.

#     The function returns a list of triples in the form of tuples, where each tuple represents an entity-relation-entity triple.
#     If the output string is not a valid JSON or does not contain the expected key, an empty list is returned.
#     """
#     try:
#         # Remove leading and trailing non-JSON parts
#         json_str_start = output.find("{")
#         json_str_end = output.rfind("}") + 1
#         clean_json_str = output[json_str_start:json_str_end]

#         data = json.loads(clean_json_str)

#         triples_list = data["entities_and_triples"]
#         entities = {}
#         triples = []

#         # Process entities
#         for item in triples_list:
#             if "," in item and ":" in item:  # Likely an entity
#                 parts = item.split(",", 1)
#                 entity_id = parts[0].strip("[] ")
#                 entity_type, entity_name = parts[1].split(":", 1)
#                 entities[entity_id] = {
#                     "type": entity_type.strip(),
#                     "name": entity_name.strip(),
#                 }

#         # Process relationships
#         for item in triples_list:
#             if (
#                 item.count("[") == 2 and item.count("]") == 2 and " " in item
#             ):  # Likely a relationship
#                 parts = item.split(" ")
#                 if len(parts) == 3:
#                     entity1_id = parts[0].strip("[]")
#                     relation = parts[1].strip()
#                     entity2_id = parts[2].strip("[]")
#                     if entity1_id in entities and entity2_id in entities:
#                         entity1_name = entities[entity1_id]["name"]
#                         entity2_name = entities[entity2_id]["name"]
#                         triples.append((entity1_name, relation, entity2_name))

#         return triples
#     except json.JSONDecodeError:
#         return []

def parse_output_to_triples(output):
    """
    Parse the output string and extract triples from it.

    Args:
        output (str): The output string to parse.

    Returns:
        list: A list of triples extracted from the output string.

    Raises:
        None

    The function takes an output string and extracts triples from it. It assumes that the output string
    contains a JSON object with a key "entities_and_triples" which holds a list of strings representing
    entities and relationships. The function iterates over the list and identifies entities and relationships
    based on certain patterns. It then constructs triples using the identified entities and relationships.

    The function returns a list of triples in the form of tuples, where each tuple represents an entity-relation-entity triple.
    If the output string is not a valid JSON or does not contain the expected key, an empty list is returned.
    """
    try:
        # Load allowed relations
        with open('src/allowed_detailled_relations.json', 'r') as f:
            allowed_relations = json.load(f)

        # Remove leading and trailing non-JSON parts
        json_str_start = output.find("{")
        json_str_end = output.rfind("}") + 1
        clean_json_str = output[json_str_start:json_str_end]

        data = json.loads(clean_json_str)

        triples_list = data["entities_and_triples"]
        entities = {}
        triples = []

        # Process entities
        for item in triples_list:
            if "," in item and ":" in item:  # Likely an entity
                parts = item.split(",", 1)
                entity_id = parts[0].strip("[] ")
                entity_type, entity_name = parts[1].split(":", 1)
                entities[entity_id] = {
                    "type": entity_type.strip(),
                    "name": entity_name.strip(),
                }

        # Process relationships
        for item in triples_list:
            if (
                item.count("[") == 2 and item.count("]") == 2 and " " in item
            ):  # Likely a relationship
                parts = item.split(" ")
                if len(parts) == 3:
                    entity1_id = parts[0].strip("[]")
                    relation = parts[1].strip()
                    entity2_id = parts[2].strip("[]")
                    if entity1_id in entities and entity2_id in entities:
                        entity1_type = entities[entity1_id]["type"]
                        entity2_type = entities[entity2_id]["type"]
                        entity1_name = entities[entity1_id]["name"]
                        entity2_name = entities[entity2_id]["name"]
                        
                        # Check if the relation is allowed
                        for allowed_relation in allowed_relations.values():
                            #print('Tuple:', (entity1_type, relation.lower(), entity2_type))
                            if (entity1_type.lower(), relation.lower(), entity2_type.lower()) == tuple(allowed_relation):
                                triples.append((entity1_name, relation, entity2_name))
                                break

        return triples
    except json.JSONDecodeError:
        return []
    except FileNotFoundError:
        print("Allowed relations file not found.")
        return []

class EntityExtractor:
    def __init__(
        self,
        config,
    ):
        self.entity_detection_threshold = config["entity_detection_threshold"]
        self.relation_extraction_threshold = config[
            "relation_extraction_threshold"
        ]

        self.entity_model = GLiNER.from_pretrained(
            config["entity_model_name"], map_location="cuda"
        )
        self.relation_model = GLiNER.from_pretrained(
            config["relation_model_name"], map_location="cuda"
        )

        allowed_entities_path = config["allowed_entities_path"]
        allowed_relations_path = config["allowed_relations_path"]
        allowed_precise_relations_path = config[
            "allowed_precise_relations_path"
        ]

        with open(allowed_entities_path, "r") as f:
            self.allowed_entities = json.load(f)

        with open(allowed_relations_path, "r") as f:
            self.allowed_relations = json.load(f)

        with open(
            allowed_precise_relations_path, "r"
        ) as f:  # Load detailed relations
            self.allowed_detailed_relations = json.load(f)
            
    

    def extract_entities(self, text):
        """
        Extracts entities from the given text using Gliner family of models (very fast).

        Args:
            text (str): The input text from which entities need to be extracted.

        Returns:
            list: A list of dictionaries representing the extracted entities.
                  Each dictionary contains the entity label and its corresponding value.
        """
        labels = list(self.allowed_entities.values())
        entities = self.entity_model.predict_entities(
            text, labels, threshold=self.entity_detection_threshold
        )
        entities = [dict(t) for t in {tuple(d.items()) for d in entities}]
        return entities

    def extract_relations(self, text: str, entities: List[dict]) -> List[tuple]:
        """
        Extracts relations between entities from the given text using pre-extracted entities.

        Args:
            text (str): The input text from which relations are to be extracted.
            entities (List[dict]): A list of dictionaries representing the entities found in the text.
                                   Each dictionary should have a "text" key containing the entity text.

        Returns:
            List[tuple]: A list of tuples representing the extracted relations.
                         Each tuple contains three elements: (entity1, relation, entity2).
        """
        list_entities = [entity["text"] for entity in entities]
        relations = list(self.allowed_relations.values())
        all_relations = []
        possible_relations = [
            f"{entity} <> {relation}"
            for entity in list_entities
            for relation in relations
        ]
        for i in range(0, len(possible_relations), 4):
            labels = possible_relations[i : i + 4]
            entities_relations = self.relation_model.predict_entities(
                text, labels, threshold=self.relation_extraction_threshold
            )
            entities_relations = [
                {
                    "e1": k["label"].split("<>")[0],
                    "relation": k["label"].split("<>")[1],
                    "e2": k["text"],
                }
                for k in entities_relations
            ]
            list_triples = [
                (k["e1"], k["relation"], k["e2"]) for k in entities_relations
            ]
            all_relations.extend(list_triples)
        all_relations = list(set(all_relations))
        return all_relations

    def extract_relations_v3(self, text):
        """
        Extracts relations from the given text using the Triplex model (SOTA but slow).

        Args:
            text (str): The input text from which relations need to be extracted.

        Returns:
            list: A list of triples representing the extracted relations.
        """
        entity_types = [
            entity_type.upper()
            for entity_type in self.allowed_entities.values()
        ]
        predicates = [
            predicate.upper() for predicate in self.allowed_relations.values()
        ]
    
        message = format_triplex_input(text, entity_types, predicates)
        messages = [{"role": "user", "content": message}]
        options = {"temperature": 0.01}
        model = "sciphi/triplex:latest"
        # load the sciphi/triplex model if not already loaded
        from src.generation_utils import pull_model

        pull_model("sciphi/triplex:latest")
        response = ollama.chat(model=model, messages=messages, options=options)
        output = response["message"]["content"]
        #print("RAW OUTPUT:", output)
        return parse_output_to_triples(output)
    
    # def extract_relations_v4(self, text: str, entities: List[dict]) -> List[tuple]:
    #     """
    #     Extracts relations from the given text using the Relik model.

    #     Args:
    #         text (str): The input text from which relations are to be extracted.
    #         entities (List[dict]): A list of dictionaries representing the entities found in the text.
    #                                Each dictionary should have a "text" key containing the entity text.

    #     Returns:
    #         List[tuple]: A list of tuples representing the extracted relations.
    #                      Each tuple contains three elements: (entity1, relation, entity2).
    #     """
    #     # Extract relations using Relik model
    #     relik_out: RelikOutput = self.relik_model(text)

    #     # Parse the output to extract relations
    #     relations = []
    #     for triplet in relik_out.triplets:
    #         subject = triplet.subject.text
    #         relation = triplet.label
    #         obj = triplet.object.text
    #         relations.append((subject, relation, obj))

    #     return relations

    @staticmethod
    def parse_output_to_triples(output):
        """
        Parse the output string and extract triples from it.

        Args:
            output (str): The output string to be parsed.

        Returns:
            list: A list of triples extracted from the output string.

        Raises:
            None

        """
        try:
            json_str_start = output.find("{")
            json_str_end = output.rfind("}") + 1
            clean_json_str = output[json_str_start:json_str_end]
            data = json.loads(clean_json_str)
            triples_list = data["entities_and_triples"]
            entities = {}
            triples = []
            for item in triples_list:
                if "," in item and ":" in item:
                    parts = item.split(",", 1)
                    entity_id = parts[0].strip("[] ")
                    entity_type, entity_name = parts[1].split(":", 1)
                    entities[entity_id] = {
                        "type": entity_type.strip(),
                        "name": entity_name.strip(),
                    }
            for item in triples_list:
                if (
                    item.count("[") == 2
                    and item.count("]") == 2
                    and " " in item
                ):
                    parts = item.split(" ")
                    if len(parts) == 3:
                        entity1_id = parts[0].strip("[]")
                        relation = parts[1].strip()
                        entity2_id = parts[2].strip("[]")
                        if entity1_id in entities and entity2_id in entities:
                            entity1_name = entities[entity1_id]["name"]
                            entity2_name = entities[entity2_id]["name"]
                            triples.append(
                                (entity1_name, relation, entity2_name)
                            )
            return triples
        except json.JSONDecodeError:
            return []

    def extract_precise_relations(self, text: str, entities: List[dict]):
        """
        Extracts the relations between entities from the given text. But, only the if the relations are valid (exist in the
        allowed precise relations list).
        """
        list_entities = [entity["text"] for entity in entities]
        list_entities_labels = [entity["label"] for entity in entities]
        # print("List entities labels:", list_entities_labels)
        relations = list(self.allowed_relations.values())
        all_relations = []

        precise_relations = list(self.allowed_detailed_relations.values())
        # print("PRECISE RELATIONS:", precise_relations)
        entity_relation_pairs = [
            [k[0], k[1]] for k in precise_relations
        ]  # exemple: [['Nevada', 'is located in'] is an allowed relation

        # create a list of possible relations
        possible_relations = [
            f"{entity} <> {relation}"
            for entity in list_entities
            for relation in relations
            if [list_entities_labels[list_entities.index(entity)], relation]
            in entity_relation_pairs
        ]
        # print("POSSIBLES RELATIONS:", possible_relations)

        # Splitting the process into batches due to model's limitation of handling up to 4 relation labels at a time
        if (
            possible_relations != []
        ):  # if there are no possible relations we skip the process
            for i in range(
                0, len(possible_relations), 4
            ):  # Processing 4 entities at a time
                labels = possible_relations[i : i + 4]

                # print("LABELS:", labels)
                entities_relations = self.relation_model.predict_entities(
                    text, labels
                )
                # print("entities_relations:",entities_relations)
                raw_list = [k for k in entities_relations]
                # print("rasw list:", raw_list)

                # entities_relations = [k['label'].split('<>')[0] +' '+ k['text'] for k in entities_relations]
                entities_relations = [
                    {
                        "e1": k["label"].split("<>")[0],
                        "relation": k["label"].split("<>")[1],
                        "e2": k["text"],
                    }
                    for k in entities_relations
                ]
                # transform this into a list of triples
                list_triples = [
                    (k["e1"], k["relation"], k["e2"])
                    for k in entities_relations
                ]

                # filtered_relations = [relation for relation in entities_relations if relation["label"].split(' <> ')[1] in relations]
                all_relations.extend(list_triples)

                # except Exception as ex:
                #     print("Error in extracting precise relations:", labels,"error:",ex)
                #     continue

            # delete all duplicates
        all_relations = list(set(all_relations))
        return all_relations


@lru_cache(maxsize=None)
def load_entity_extractor(config_name):
    """
    Load and initialize an EntityExtractor object based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the EntityExtractor.

    Returns:
        EntityExtractor: An instance of the EntityExtractor class.

    """

    # load the config/config.yaml file with safe_load
    with open(config_name) as file:
        config = yaml.safe_load(file)

    entity_extractor = EntityExtractor(config=config)
    return entity_extractor


def text_distance_criterion(subject, node, disambiguate_threshold):
    """
    Determines if the text distance between the subject and node is greater than the disambiguate_threshold.

    Parameters:
    subject (str): The subject text.
    node (str): The node text.
    disambiguate_threshold (int): The minimum ratio required for disambiguation.

    Returns:
    bool: True if the text distance is greater than the disambiguate_threshold, False otherwise.
    """

    value = fuzz.token_set_ratio(subject, node) > disambiguate_threshold

    return value


def cosine_sim_entity(e1, e2, model):
    """
    Calculates the cosine similarity between two entities.

    Parameters:
    e1 (str): The first entity.
    e2 (str): The second entity.
    model: The model used to encode the entities.

    Returns:
    float: The cosine similarity between the two entities.
    """
    e1_embedding = model.encode(e1, show_progress_bar=False).reshape(1, -1)
    e2_embedding = model.encode(e2, show_progress_bar=False).reshape(1, -1)
    return cosine_similarity(e1_embedding, e2_embedding)[0][0]


def advanced_disambiguation(self, subject, obj, model):
    """
    Perform modified disambiguation on the given subject and object using the specified model.

    Args:
        subject (str): The subject to disambiguate.
        obj (str): The object to disambiguate.
        model: The model to use for disambiguation.

    Returns:
        tuple: A tuple containing two lists - subject_merge_targets and object_merge_targets.
            - subject_merge_targets (list): A list of nodes in the graph that are potential merge targets for the subject.
            - object_merge_targets (list): A list of nodes in the graph that are potential merge targets for the object.
    """
    subject_merge_targets = [
        node
        for node in self.graph.nodes()
        if cosine_sim_entity(subject, node, model) > 0.85
        and fuzz.token_set_ratio(subject, node) > self.disambiguate_threshold
    ]
    object_merge_targets = [
        node
        for node in self.graph.nodes()
        if cosine_sim_entity(obj, node, model) > 0.85
        and fuzz.token_set_ratio(obj, node) > self.disambiguate_threshold
    ]
    return subject_merge_targets, object_merge_targets


class KnowledgeGraphBuilder:
    """
    Represents a knowledge graph that stores and manages knowledge triples.

    Attributes:
        triplets (List[tuple]): A list of knowledge triples in the form of (subject, predicate, object).
        graph (nx.DiGraph): A directed graph representation of the knowledge graph.

    Methods:
        __init__(self, triplets: Optional[List[tuple]] = None): Initializes a KnowledgeGraph object.
        add_triplet(self, triplet: tuple): Adds a knowledge triplet to the graph.
        generate_from_text(self, text: str): Generates knowledge triples from the given text.
        save_figure(self): Saves a visual representation of the knowledge graph as an image.
        get_triplets(self): Returns the list of knowledge triples.
        graph_to_dict(self): Converts the knowledge graph to a dictionary representation.

    """

    def __init__(
        self,
        config,
    ):
        """
        Initializes a KnowledgeGraph object.

        Args:
            triplets (Optional[List[tuple]]): A list of knowledge triples in the form of (subject, predicate, object).
                Defaults to None.

        """

        self.config = config
        self.triplets = []
        self.graph = nx.DiGraph()  # nx.Graph() #nx.DiGraph()
        for triplet in self.triplets:
            self.add_triplet(triplet)
        self.model_name = config["model_name"]
        self.llm_provider = config["llm_provider"]
        self.total_triplets = self.triplets
        self.disambiguate_threshold = config["disambiguate_threshold"]
        self.entity_extractor = EntityExtractor(config)
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device="cuda"
        )
        self.embedding_cache = {}
        self.entity_descriptions = {}
        self.community_map = {}

    def ensure_embeddings_cached(self, entities):
        missing_entities = [
            entity for entity in entities if entity not in self.embedding_cache
        ]
        if missing_entities:
            missing_embeddings = self.embedding_model.encode(
                missing_entities
            )  # , show_progress_bar=False)
            for entity, embedding in zip(missing_entities, missing_embeddings):
                self.embedding_cache[entity] = embedding.reshape(1, -1)

    # def advanced_disambiguation(self, subject, obj):
    #     """
    #     Perform modified disambiguation on the knowledge graph.

    #     Args:
    #         subject (str): The subject for disambiguation.
    #         obj (str): The object for disambiguation.

    #     Returns:
    #         tuple: A tuple containing two lists. The first list contains the merge targets for the subject,
    #                and the second list contains the merge targets for the object.
    #     """
    #     nodes = list(self.graph.nodes())
    #     self.ensure_embeddings_cached(nodes + [subject, obj])

    #     # Retrieve all embeddings from cache
    #     all_embeddings = np.vstack(
    #         [self.embedding_cache[node] for node in nodes]
    #     )
    #     subject_embedding = self.embedding_cache[subject]
    #     obj_embedding = self.embedding_cache[obj]

    #     # Compute cosine similarities in a batch
    #     subject_similarities = cosine_similarity(
    #         subject_embedding, all_embeddings
    #     ).flatten()
    #     obj_similarities = cosine_similarity(
    #         obj_embedding, all_embeddings
    #     ).flatten()

    #     # Filter nodes based on similarity and fuzzy token set ratio
    #     subject_merge_targets = [
    #         nodes[i]
    #         for i in range(len(nodes))
    #         if subject_similarities[i] > 0.7
    #         and fuzz.token_set_ratio(subject, nodes[i])
    #         > self.disambiguate_threshold
    #     ]
    #     object_merge_targets = [
    #         nodes[i]
    #         for i in range(len(nodes))
    #         if obj_similarities[i] > 0.7
    #         and fuzz.token_set_ratio(obj, nodes[i]) > self.disambiguate_threshold
    #     ]

    #     return subject_merge_targets, object_merge_targets

    def advanced_disambiguation(self, entity, use_embedding=True):
        """
        Perform modified disambiguation on the knowledge graph.

        Args:
            entity (str): The entity for disambiguation.
            use_embedding (bool): Whether to use semantic (embedding) criteria along with lexical criteria.

        Returns:
            list: A list containing the merge targets for the entity.
        """
        nodes = list(self.graph.nodes())

        if use_embedding:
            self.ensure_embeddings_cached(nodes + [entity])
            all_embeddings = np.vstack(
                [self.embedding_cache[node] for node in nodes]
            )
            entity_embedding = self.embedding_cache[entity]
            similarities = cosine_similarity(
                entity_embedding, all_embeddings
            ).flatten()

        merge_targets = [
            nodes[i]
            for i in range(len(nodes))
            if (not use_embedding or similarities[i] > 0.7)
            and fuzz.token_set_ratio(entity, nodes[i])
            > self.disambiguate_threshold
        ]

        return merge_targets

    def clean_string(self, string):
        # Remove all characters that are not letters, digits, a single space, or accented characters
        cleaned_string = re.sub(r"[^a-zA-Z0-9\s\u00C0-\u00FF]", "", string)
        # Replace multiple spaces with a single space
        cleaned_string = re.sub(r"\s+", " ", cleaned_string)
        return str(cleaned_string.strip())

    # def add_triplet(self, triplet: tuple, disambiguation=True):
    #     """
    #     Adds a knowledge triplet to the graph.

    #     Args:
    #         triplet (tuple): A knowledge triplet in the form of (subject, predicate, object).
    #     """

    #     if len(triplet) == 3:  # we keep only the triplets with 3 elements
    #         subject, predicate, obj = map(self.clean_string, triplet)
    #         self.graph.add_node(subject, entity_name=subject)
    #         self.graph.add_node(obj, entity_name=obj)
    #         self.graph.add_edge(subject, obj, label=predicate)

    #         if disambiguation:
    #             subject_merge_targets = [
    #                 node
    #                 for node in self.graph.nodes()
    #                 if fuzz.token_set_ratio(subject, node)
    #                 > self.disambiguate_threshold
    #             ]
    #             object_merge_targets = [
    #                 node
    #                 for node in self.graph.nodes()
    #                 if fuzz.token_set_ratio(obj, node) > self.disambiguate_threshold
    #             ]

    #             # model = self.embedding_model

    #             # subject_merge_targets, object_merge_targets = self.advanced_disambiguation(subject, obj)

    #             # if set(subject_merge_targets) == set(object_merge_targets):
    #             #     object_merge_targets = []

    #             for node in subject_merge_targets:
    #                 try:
    #                     if node in self.graph.nodes():
    #                         self.graph = nx.contracted_nodes(
    #                             self.graph,
    #                             subject,
    #                             node,
    #                             self_loops=False,
    #                             copy=True,
    #                         )
    #                 except Exception as ex:
    #                     print(
    #                         f"Error in triplet: {triplet} while merging subject with entity: {node}"
    #                     )

    #             for node in object_merge_targets:
    #                 try:
    #                     if node in self.graph.nodes():
    #                         self.graph = nx.contracted_nodes(
    #                             self.graph,
    #                             obj,
    #                             node,
    #                             self_loops=False,
    #                             copy=True,
    #                         )
    #                 except Exception as ex:
    #                     print(
    #                         f"Error in triplet: {triplet} while merging object with entity: {node}"
    #                     )

    def add_triplet(self, triplet: tuple, disambiguation=True):
        """
        Adds a knowledge triplet to the graph.

        Args:
            triplet (tuple): A knowledge triplet in the form of (subject, predicate, object).
        """

        if len(triplet) == 3:  # we keep only the triplets with 3 elements
            subject, predicate, obj = map(self.clean_string, triplet)
            self.graph.add_node(subject, entity_name=subject)
            self.graph.add_node(obj, entity_name=obj)
            self.graph.add_edge(subject, obj, label=predicate)


    def disambiguate_entities(self):
        """
        Disambiguates entities in the knowledge graph.
        This method iterates over the nodes in the graph and performs entity disambiguation.
        It identifies merge targets for each node and merges them if necessary.
        The merge is performed by contracting nodes in the graph.
        Returns:
            None
        Raises:
            Exception: If an error occurs while merging nodes.
        """
    
        processed_nodes = set()
        nodes = list(self.graph.nodes())
        # self.ensure_embeddings_cached(nodes)

        for node in nodes:
            if node in processed_nodes:
                continue

            # Get merge targets using modified disambiguationsubject_merge_targets, _ = self.advanced_disambiguation(node, node)

            # subject_merge_targets, _ = self.advanced_disambiguation(node, node)
            subject_merge_targets = self.advanced_disambiguation(
                node, use_embedding=False
            )

            if subject_merge_targets:
                subject_merge_targets.append(node)
                highest_degree_node = max(
                    subject_merge_targets, key=lambda n: self.graph.degree(n)
                )

                # Print the list of similar entities and the entity with the highest degree
                # print(f"Similar entities for '{node}': {subject_merge_targets}, Highest degree entity: '{highest_degree_node}'")

                for similar_node in subject_merge_targets:
                    if (
                        similar_node != highest_degree_node
                        and similar_node in self.graph.nodes()
                    ):
                        try:
                            self.graph = nx.contracted_nodes(
                                self.graph,
                                highest_degree_node,
                                similar_node,
                                self_loops=False,
                                copy=True,
                            )
                            processed_nodes.add(similar_node)
                        except Exception as ex:
                            print(
                                f"Error while merging node: {similar_node} with entity: {highest_degree_node}"
                            )

    def complete_from_text(self, text: str):
        """
        Generates knowledge triples from the given text.

        Args:
            text (str): The input text from which knowledge triples are extracted.

        """
        document = text
        prompt = f"""
            You are an AI assistant expert in relation extraction whose goal is to extract all the relational triples from a text.
            A knowledge triple is a clause that contains a subject, a predicate,
            and an object. The subject is the entity being described,
            the predicate is the property of the subject that is being
            described, and the object is the value of the property. If you can build relationships between the existing knowledge graph and the new text, do so, if you spot a missing relationship in the current knowledge graph, add it to the list.

            EXAMPLE
            It's a state in the US. It's also the number 1 producer of gold in the US.

            Output: {{
                "list_relations": [
                    {{
                        "subject": "Nevada",
                        "predicate": "is a",
                        "object": "state"
                    }},
                    {{
                        "subject": "Nevada",
                        "predicate": "is in",
                        "object": "US"
                    }},
                    {{
                        "subject": "Nevada",
                        "predicate": "is the number 1 producer of",
                        "object": "gold"
                    }}
                ]
            }}
            END OF EXAMPLE

            EXAMPLE
            Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.

            Output: {{
                "list_relations": [
                    {{
                        "subject": "Descartes",
                        "predicate": "likes to drive",
                        "object": "antique scooters"
                    }},
                    {{
                        "subject": "Descartes",
                        "predicate": "plays",
                        "object": "mandolin"
                    }}
                ]
            }}
            END OF EXAMPLE

            
            Here is the text to process:
            
            {document}
            Output:
        """

        class RelationList(BaseModel):
            list_relations: List[dict] = Field(
                description="List of relations as knowledge triples of strings: {'subject': subject, 'predicate': predicate, 'object': object}"
            )

        result = LLM_answer_v3(
            prompt=prompt,
            llm_provider=self.llm_provider,
            model_name=self.model_name,
            temperature=1,
            stream=False,
            pydantic_object=RelationList,
            json_formatting=True,
        )

        # Parse the LLM response
        print("RAW RESPONSE:", result)
        triplets = result["list_relations"]
        print("TRIPLETS:", triplets)
        # convert list of dicts to list of tuples
        list_tuples = []
        for triplet in triplets:
            try:
                subject = triplet["subject"]
                predicate = triplet["predicate"]
                obj = triplet["object"]
                list_tuples.append((subject, predicate, obj))
            except:
                print("Error in triplet:", triplet, "Skipping...")

        self.total_triplets.append(list_tuples)

        for triplet in list_tuples:  # we add the new triplets to the graph and proced to entity resolution/disambiguation
            self.add_triplet(triplet)

    def complete_from_text_v2(self, text: str):
        """
        Extract knowledge triples from the given text and complete the knowledge graph.

        Args:
            text (str): The input text from which knowledge triples are extracted.

        """

        # Extract entities from the text
        entities = self.entity_extractor.extract_entities(text)
        # print('ENTITIES:', entities)

        # Extract relations based on the entities found
        # relations = self.entity_extractor.extract_relations(text, entities)
        relations = self.entity_extractor.extract_precise_relations(
            text, entities
        )
        # print("ALLL RELATIONS:", relations)

        # Add the extracted relations as triplets to the knowledge graph
        for relation in relations:
            self.add_triplet(relation, disambiguation=True)

    def complete_from_text_v3(self, text: str):
        """
        Extract knowledge triples from the given text and complete the knowledge graph.
        This version use Triplex model for extraction.
        """
        # Extract relations based on the entities found
        relations = self.entity_extractor.extract_relations_v3(text)
        # Add the extracted relations as triplets to the knowledge graph
        #print("RELATIONS:", relations)
        for relation in relations:
            self.add_triplet(relation, disambiguation=True)

    def complete_from_text_v3_batch(self, texts: list):
        """
        Extract knowledge triples from a list of texts and complete the knowledge graph in parallel.
        This version uses the Triplex model for extraction.
        """
        import threading

        def process_text(text):
            self.complete_from_text_v3(text)

        threads = []
        for text in texts:
            thread = threading.Thread(target=process_text, args=(text,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def get_triplets(self):
        """
        Returns the list of knowledge triples.

        Returns:
            List[tuple]: A list of knowledge triples in the form of (subject, predicate, object).

        """
        return self.triplets

    def graph_to_dict(self):
        """
        Converts the knowledge graph to a dictionary representation.

        Returns:
            dict: A dictionary representation of the knowledge graph.

        """
        if self.triplets != []:
            G = {}
            for triplet in self.triplets:
                try:
                    subject, predicate, obj = triplet
                    if subject in G:
                        if predicate in G[subject]:
                            G[subject][predicate].append(obj)
                        else:
                            G[subject][predicate] = [obj]
                    else:
                        G[subject] = {predicate: [obj]}
                except Exception as ex:
                    print("Exception:", ex)
                    print("Error in triplet:", triplet, "Skipping...")
        else:
            G = {"Empty graph": "Knowledge graph is empty for now"}

        return G

    def retrieve_linked_entities(self, node: str):
        """
        Retrieves all entities linked with the given node.

        Args:
            node (str): The node for which to retrieve linked entities.

        Returns:
            List[tuples]: A list of entities linked with the given node. format: triplets (e1, relation, e2).
        """

        if node not in self.graph:
            print(f"Node {node} does not exist in the graph.")
            return []

        linked_entities_and_relations = []
        # Retrieve successors and the relations (labels) to them
        for successor in self.graph.successors(node):
            # Access the 'label' attribute for the edge
            relation_label = self.graph[node][successor]["label"]
            linked_entities_and_relations.append(
                (node, relation_label, successor)
            )
        # Retrieve predecessors and the relations (labels) from them
        for predecessor in self.graph.predecessors(node):
            # Access the 'label' attribute for the edge
            relation_label = self.graph[predecessor][node]["label"]
            linked_entities_and_relations.append(
                (predecessor, relation_label, node)
            )

        return linked_entities_and_relations

    def detect_communities_v2(self):
        """
        Detects communities in the graph using the Leiden algorithm and returns a tuple containing
        the community map (a dictionary mapping each node to its community) and the community colors
        (a list of color codes for each community).

        Returns:
            tuple: A tuple containing:
                - community_map (dict): A dictionary where keys are node identifiers and values are community identifiers.
                - community_colors (list): A list of hexadecimal color codes for the communities.
        """
        import colorsys

        import igraph as ig
        import leidenalg
        import networkx as nx

        # Convert the NetworkX graph to an igraph graph
        g_igraph = ig.Graph.TupleList(self.graph.edges(), directed=False)
        g_igraph.vs["name"] = list(self.graph.nodes())

        # Use the Leiden algorithm to detect communities #leidenalg.ModularityVertexPartition,
        partition = leidenalg.find_partition(
            g_igraph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=2,
        )

        # Create a dictionary to map each node to its community
        community_map = {}
        for i, comm in enumerate(partition):
            for node in comm:
                community_map[g_igraph.vs[node]["name"]] = i

        # Define a color palette for the communities
        community_colors = [
            "#{:02x}{:02x}{:02x}".format(
                *(
                    int(
                        colorsys.hsv_to_rgb(i / len(partition), 1, 0.7)[j] * 255
                    )
                    for j in range(3)
                )
            )
            for i in range(
                len(partition) + 1
            )  # plus 1 because we add the isolated nodes to the last community
        ]
        # Après avoir créé community_map
        # Après avoir créé community_map
        isolated_nodes = set(self.graph.nodes()) - set(community_map.keys())
        # Trouver le dernier numéro de communauté utilisé
        last_community_num = max(community_map.values()) + 1
        for node in isolated_nodes:
            community_map[node] = (
                last_community_num  # Attribuer le même numéro à tous les nœuds isolés
            )

        # print("COMMUNITY MAP:", community_map)
        return community_map, community_colors

    def detect_communities(self):
        """
        Detects communities in the graph using the Louvain algorithm and returns a tuple containing
        the community map (a dictionary mapping each node to its community) and the community colors
        (a list of color codes for each community).

        Returns:
            tuple: A tuple containing:
                - community_map (dict): A dictionary where keys are node identifiers and values are community identifiers.
                - community_colors (list): A list of hexadecimal color codes for the communities.
        """
        import colorsys

        from networkx.algorithms import community

        # Use the Louvain algorithm to detect communities
        communities = community.louvain_communities(self.graph, seed=42)

        # Create a dictionary to map each node to its community
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i

        # Define a color palette for the communities
        community_colors = [
            "#{:02x}{:02x}{:02x}".format(
                *(
                    int(
                        colorsys.hsv_to_rgb(i / len(communities), 1, 0.7)[j]
                        * 255
                    )
                    for j in range(3)
                )
            )
            for i in range(len(communities))
        ]

        # print("COMMUNITY MAP:", community_map)

        return community_map, community_colors

    @lru_cache(maxsize=None)
    def get_description_dict(self):
        """
        Return a dictionnary where the keys are the nodes and the values are the description of the nodes.
        Args:
            None
        Returns:
            dict: A dictionary where keys are node identifiers and values are the description of the nodes.
        """

        # get all the nodes
        nodes = list(self.graph.nodes())  # types list of strings ?
        # initialize the dictionary
        description_dict = {}
        # loop over the nodes
        for node in tqdm(nodes, desc="Extracting descriptions of entities ..."):
            # get the description of the node using entity_description function
            description = entity_description(node, default_config=self.config)
            # add the description to the dictionary
            description_dict[node] = description

        return description_dict

    def summarize_community(self, community_prompt):
        """
        Takes a community description as input: entities + entity descriptions + relations and returns a summary of the community.
        """
        summarization_prompt = f"""Write a short summary of the following community of entities of a knowledge graph. 
        The content of this summary includes an overview of the community's key entities and relationships.\n # Report Structure \n
        The report should include the following sections:\n
        - TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.\n
        - SUMMARY: An summary of the community's overall structure and goal, how its entities are related to each other, and significant points associated with its entities.
        
        Here is the community to summarize: \n\n
        
        {community_prompt}
        """

        community_report = LLM_answer_v3(
            prompt=summarization_prompt,
            llm_provider=self.config["description_llm_provider"],
            model_name=self.config["description_model_name"],
            temperature=1,
            stream=False,
        )

        return community_report

    @lru_cache(maxsize=None)
    def get_communities_summaries(self):
        """
        Gather all the entities grouped by communities along with each entity's description. And for each community, makes a summary
        explaining the content of the community.

        Args:
            None
        Returns:
            dict: A dictionary where keys are community identifiers (ints) and values are dictionaries containing the community's description.
        """

        # get the community map dictionary
        community_map, community_colors = self.detect_communities_v2()
        self.community_map = community_map
        # print("CURRENT COMMUNITY MAP:", community_map)
        # get the description dictionary
        description_dict = self.get_description_dict()

        # initialize the dictionary which will map the key of the community to community summary
        communities_prompts = {}

        # build a dictionary that maps community identifiers to lists of entities
        community_entities = {}
        for entity, community_id in community_map.items():
            if community_id not in community_entities:
                community_entities[community_id] = []
            community_entities[community_id].append(entity)

        # generate prompts for each community
        for community_id, entities in community_entities.items():
            prompt = "List of entities in the community:"
            for entity in entities:
                prompt += f"\n- {entity} : {description_dict.get(entity, 'No description available')}"

            # add all relation triples for the community (all relations of each entity in the community)
            prompt += "\nList of relations in the community:"
            for entity in entities:
                for relation in self.retrieve_linked_entities(entity):
                    prompt += f"\n- {str(relation)}"

            communities_prompts[community_id] = prompt

        # print the prompts
        for community_id, prompt in communities_prompts.items():
            print(f"PROMPT for community {community_id}:", prompt)

        # We create a summary for each community
        for community_id, community_prompt in tqdm(
            communities_prompts.items(), desc="Summarizing communities ..."
        ):
            community_summary = self.summarize_community(community_prompt)
            # add a key to the dictionary with the community id and the summary as value
            communities_prompts[community_id] = community_summary

        return communities_prompts

    def plot_graph(self, community_map=None, community_colors=None):
        """
        Plots the graph using Plotly. If community_map and community_colors are provided,
        nodes are colored by their community. Otherwise, nodes are colored by their degree.

        Args:
            community_map (dict, optional): A dictionary mapping each node to its community. Defaults to None.
            community_colors (list, optional): A list of color codes for the communities. Defaults to None.
        """
        import colorsys

        import networkx as nx
        import plotly.graph_objs as go

        # Calculate node positions with the spring layout in 3D
        pos = nx.spring_layout(self.graph, dim=3)

        # Initialize lists for edge and node coordinates
        edge_x, edge_y, edge_z = [], [], []
        node_x, node_y, node_z = [], [], []
        node_info = []  # To store the name of the entity associated with each node

        if community_map is None or community_colors is None:
            # Calculate node degrees for color mapping
            degrees = nx.degree(self.graph)
            max_degree = max(dict(degrees).values())
            # Map each node's degree to a color
            degree_colors = {
                node: "#{:02x}{:02x}{:02x}".format(
                    *(
                        int(
                            colorsys.hsv_to_rgb(degree / max_degree, 1, 0.7)[j]
                            * 255
                        )
                        for j in range(3)
                    )
                )
                for node, degree in degrees
            }

        # Loop to add edge coordinates
        for edge in self.graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # Add the x coordinates of the edge
            edge_y.extend([y0, y1, None])  # Add the y coordinates of the edge
            edge_z.extend([z0, z1, None])  # Add the z coordinates of the edge

        # Create a trace for the edges in 3D
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line=dict(width=0.5, color="#888"),  # Use a neutral color for edges
            hoverinfo="none",
            mode="lines",
        )

        # Loop to add node coordinates and info
        for node in self.graph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_info.append(
                f"{node}"
            )  # Add the name of the entity as info to display

        # Create a trace for the nodes in 3D
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text",  # Changed from "markers" to "markers+text" to display labels permanently
            hoverinfo="text",
            text=node_info,
            textposition="top center",  # Position the text above the markers
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                reversescale=True,
                color=[
                    community_colors[community_map[node]]
                    if community_map and community_colors
                    else degree_colors[node]
                    for node in self.graph.nodes()
                ],
                size=10,
                colorbar=dict(thickness=15, title="Node Connections"),
                line_width=2,
            ),
        )

        # Adjust node size based on the number of connections, with a minimum size
        node_adjacencies = []
        for node, adjacencies in enumerate(self.graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        node_trace.marker.size = [
            5 + len(adjacencies[1]) * 2
            for node, adjacencies in enumerate(self.graph.adjacency())
        ]  # Adjust size here

        # Create the figure with 3D node and edge traces
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<br>Interactive Knowledge Graph in 3D",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                scene=dict(
                    xaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False
                    ),
                    yaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False
                    ),
                    zaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False
                    ),
                ),
            ),
        )

        # Display the figure
        fig.show()

    def interactive_graph(self, communities=False):
        """
        Generates an interactive graph. If communities is True, it detects communities
        and colors the nodes accordingly. Otherwise, it colors nodes based on their degree.

        Args:
            communities (bool, optional): Whether to detect and color nodes by communities. Defaults to False.
        """
        if communities:
            community_map, community_colors = self.detect_communities_v2()
            self.plot_graph(community_map, community_colors)
        else:
            self.plot_graph()

    def get_graph(self):
        return self.graph


class KnowledgeGraphIndex:
    def __init__(self, config):
        self.entity_index = {}
        self.disambiguate_threshold = config["disambiguate_threshold"]
        self.config = config
        self.builder = KnowledgeGraphBuilder(config)
        self.nx_graph = self.builder.get_graph()

    def from_documents(
            self, documents: List[Document], overwrite: bool = True
        ) -> KnowledgeGraph:
            """
            Creates a knowledge graph from a list of documents.
    
            Args:
                documents (List[str]): A list of text documents.
    
            Returns:
                KnowledgeGraph: A knowledge graph containing the extracted knowledge triples.
    
            """
    
            # We create a new KnowledgeGraph object
            # We go through each document and extract knowledge triples from the text
            for document in tqdm(documents, desc="Extracting knowledge relations"):
                self.builder.complete_from_text_v2(text=document.page_content)
    
            # We lauch the final node disambiguation
            self.builder.disambiguate_entities()
    
            # BATCH_SIZE = 4
            # #same thing but using complete_from_text_v3 and batched requests (complete from text v3 takes a list of texts as input)
            # for i in tqdm(range(0, len(documents), BATCH_SIZE)):
            #     batch = documents[i : i + BATCH_SIZE]
            #     texts = [doc.page_content for doc in batch]
            #     self.builder.complete_from_text_v3_batch(texts)
    
            # We get the final knowledge graph object
            knowledge_graph = self.builder.get_graph()
    
            # We add the knowledge graph to the class attribute
            self.nx_graph = knowledge_graph
    
            # We extract all the entities from the knowledge graph
            entities = list(knowledge_graph.nodes())
    
            print("LIST EXRTRACTED Entities:", entities)
            print("NUMBER OF EXTRACTED ENTITIES:", len(entities))
    
            # we create a dictionnary matching each entity to a specific index
            self.entity_index = {
                entity: index for index, entity in enumerate(entities)
            }
    
            # We use the function get_description_dict to get the description of each entity/nodes
            description_dict = self.builder.get_description_dict()
    
            # get the community summaries
            community_summaries = self.builder.get_communities_summaries()
    
            # we get the community_map
            community_map = self.builder.community_map
            
            # We create a list of all the unique relations labels
            unique_relations = list(
                set(
                    [
                        knowledge_graph.edges[edge]["label"]
                        for edge in knowledge_graph.edges()
                    ]
                )
            )
     
            # we create a dictionnary matching each relation to a specific index
            self.relation_index = {
                relation: index for index, relation in enumerate(unique_relations)
            }
    
            # For each entity we create a langchain Document object containing the entity, the relations and the linked entities as metadata and its id
            list_documents = []
            for entity in tqdm(
                entities, desc="Creating Qdrant Knowledge Graph Store..."
            ):
                linked_entities = list(knowledge_graph.successors(entity)) + list(
                    knowledge_graph.predecessors(entity)
                )
                relation_list = []
                for linked_entity in linked_entities:
                    try:
                        # Try to get the relation in the forward direction
                        relation = knowledge_graph[entity][linked_entity]["label"]
                    except KeyError:
                        try:
                            # If the forward direction fails, try the reverse direction
                            relation = knowledge_graph[linked_entity][entity][
                                "label"
                            ]
                        except KeyError:
                            
                            relation = None  
                    if relation is not None:
                        relation_list.append(relation)
    
                description = description_dict[entity]
    
                #Add the entity to the list_documents
                document = Document(
                    page_content=entity,
                    metadata={
                        "relation_list": [
                            {"relation": relation, "linked_entity": linked_entity}
                            for relation, linked_entity in zip(
                                relation_list, linked_entities
                            )
                        ],
                        "description": description,
                        "type": "entity",
                        "community": {
                            "id": community_map[entity],
                            "summary": community_summaries[community_map[entity]],
                        },
                    },
                )
    
                list_documents.append(document)
    
            # Add entity descriptions to the list_documents
            for entity, description in description_dict.items():
                document = Document(
                    page_content=description,
                    metadata={
                        "type": "entity_description",
                        "entity": entity,
                    },
                )
                list_documents.append(document)
    
            # Add community summaries to the list_documents
            for community_id, summary in community_summaries.items():
                document = Document(
                    page_content=summary,
                    metadata={
                        "type": "community_summary",
                        "community_id": community_id,
                    },
                )
                list_documents.append(document)
    
            embedding_model = get_embedding_model(
                model_name=self.config["embedding_model"]
            )
    
            # delete the previous vectorstore if it exists
            if (
                os.path.exists(self.config["persist_directory"] + "_kg")
                and overwrite
            ):
                shutil.rmtree(self.config["persist_directory"] + "_kg")
    
            vectordb = Qdrant.from_documents(
                documents=list_documents,
                embedding=embedding_model,
                path=self.config["persist_directory"] + "_kg",
                collection_name="qdrant_vectorstore",
            )
    
            return vectordb

    def query_graph(self, query: str):
        """
        Queries the knowledge graph with a given query using the cypher query language.

        Args:
            query (str): The query to be executed on the knowledge graph.

        Returns:
            List[dict]: A list of dictionaries containing the query results.
        """
        # we print the list of nodes of the graph
        print("NODES:", self.nx_graph.nodes())
        query_result = GrandCypher(self.nx_graph).run(query)
        return query_result

    def get_graph(self):
        return self.nx_graph

    def extract_subgraph(self, entities):
        graph = self.nx_graph
        subgraph_nodes = set(entities)  # Use a set to avoid duplicates

        # Add neighbors for each entity
        for entity in entities:
            subgraph_nodes.update(graph.neighbors(entity))

        # Extract the subgraph
        subgraph = graph.subgraph(subgraph_nodes)

        # Convert the subgraph to a dictionary of dictionaries manually, ensuring only connected nodes are included
        dict_graph = {}
        for u in subgraph_nodes:
            # Filter out 'contraction' attribute from the edge data
            neighbors = {
                v: {
                    key: value
                    for key, value in data.items()
                    if key != "contraction"
                }
                for v, data in subgraph[u].items()
                if v in subgraph_nodes
            }
            if neighbors:  # Only include nodes that have connections
                dict_graph[u] = neighbors

        return dict_graph

