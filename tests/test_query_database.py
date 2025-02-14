


from src.main_utils.retrieval_utils_v2 import RetrievalAgent
import yaml


if __name__=='__main__':
    
    with open("config/test_config.yaml") as f:
        config = yaml.safe_load(f)

    agent = RetrievalAgent(config)

    # Query the database
    query = "Qui doit assumer face à la France périphérique le discours de Mélenchon?"
    compressed_docs = agent.query_database(query)
    print("Compressed documents:", compressed_docs)