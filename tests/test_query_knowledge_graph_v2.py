import yaml

from src.knowledge_graph import load_entity_extractor
from src.retrieval_utils import query_knowledge_graph_v2

if __name__ == "__main__":
    # load the config/config.yaml file with safe_load

    # load config from config/config.yaml
    with open("config/config.yaml") as file:
        config = yaml.safe_load(file)

    query = "Quel est le lien entre robin et Tang ?"

    entity_extractor = load_entity_extractor("config/config.yaml")
    entities = entity_extractor.extract_entities(query)
    print("ENTITIES:", entities)

    nodes = query_knowledge_graph_v2(query=query, default_config=config)
    print("NODES:", nodes)
