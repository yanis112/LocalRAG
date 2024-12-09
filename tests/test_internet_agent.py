import pytest
import yaml
import os
import shutil
from src.main_utils.generation_utils_v2 import RAGAgent

@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture
def agent(config):
    return RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})

def test_internet_rag(agent):
    query = "Is it true that Elon Musk offered money to Americans to vote for Trump explicitly?"

    # Ensure the internet folder is empty
    if os.path.exists("data/internet"):
        shutil.rmtree("data/internet")
    os.makedirs("data/internet")

    # Call the method
    answer = agent.internet_rag(query)
    
    print("Answer: ", answer)

    # Assertions
    assert isinstance(answer, str), "The answer should be a string"

if __name__ == "__main__":
    pytest.main()