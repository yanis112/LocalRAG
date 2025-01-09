from src.main_utils.vectorstore_utils_v4 import VectorAgent
import yaml

if __name__=="__main__":
    #load config from config/config.yaml
    with open('config/config.yaml','r') as f:
        config=yaml.safe_load(f)    
    config["path"]='test_data'
    config["persist_directory"]="data/vector_stores/qdrant_semantic_test"
    agent = VectorAgent(default_config=config)
    agent.fill()
    
    #try deleting food docs from vectorstore
    agent.delete(folders=["test_data/sub1"])
    
    