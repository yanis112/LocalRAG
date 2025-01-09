import pytest
import time
import yaml
from src.main_utils.generation_utils_v2 import RAGAgent

@pytest.fixture
def config():
    """Fixture to load configuration file."""
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture
def rag_agent(config):
    """Fixture to create RAGAgent instance."""
    return RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})

def test_rag_agent_initialization(rag_agent):
    """Test RAGAgent initialization."""
    assert isinstance(rag_agent, RAGAgent)

@pytest.mark.parametrize("query,expected_type", [
    ("Est il vrai que Elon Musk a proposé de l'argent à des américains pour qu'ils votent pour Trump explicitement ?", str),
])
def test_advanced_rag_answer(rag_agent, query, expected_type, caplog):
    """Test RAG answer generation with timing and output verification."""
    start_time = time.time()
    
    # Generate answer
    response = rag_agent.advanced_RAG_answer(query)
    
    end_time = time.time()
    execution_time = end_time - start_time

    # Assertions
    assert isinstance(response, expected_type)
    assert len(response) > 0
    assert execution_time > 0

    # Print results to console
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])