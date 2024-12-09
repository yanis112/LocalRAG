import pytest
import yaml
from src.main_utils.generation_utils_v2 import RAGAgent

@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture
def agent(config):
    return RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})

def test_RAG_answer(agent):
    query = "Who is Simon Boiko?"
    answer = agent.RAG_answer(query)
    assert isinstance(answer, str)

def test_advanced_RAG_answer(agent):
    query = "Who is Simon Boiko?"
    answer = agent.advanced_RAG_answer(query)
    assert isinstance(answer, str)

def test_internet_rag(agent):
    query = "Is it true that Elon Musk offered money to Americans to vote for Trump explicitly?"
    answer, _ = agent.internet_rag(query)
    assert isinstance(answer, str)