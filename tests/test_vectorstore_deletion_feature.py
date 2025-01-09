import sys
import os
import pytest
import yaml
from src.main_utils.vectorstore_utils_v4 import VectorAgent

@pytest.fixture
def vector_agent():
    sys.stdout.write("\n=== Setting up test environment ===\n")
    sys.stdout.flush()
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config.update({
        "path": "test_data",
        "persist_directory": "data/vector_stores/qdrant_semantic_test"
    })
    
    # Create test agent
    agent = VectorAgent(default_config=config)
    
    yield agent
    
    # Cleanup
    sys.stdout.write("\n=== Cleaning up test environment ===\n")
    sys.stdout.flush()
    if os.path.exists("test_data"):
        import shutil
        shutil.rmtree("test_data")

class TestVectorAgent:
    def test_file_creation(self, vector_agent):
        sys.stdout.write("\n=== Testing file creation ===\n")
        sys.stdout.flush()
        
        # Create test directory and file
        os.makedirs("test_data/sub1", exist_ok=True)
        with open("test_data/sub1/fake.txt", "w") as f:
            f.write("fake")
            
        assert os.path.exists("test_data/sub1/fake.txt")
        sys.stdout.write("File creation successful\n")
        sys.stdout.flush()

    def test_vector_store_fill(self, vector_agent):
        sys.stdout.write("\n=== Testing vector store fill ===\n")
        sys.stdout.flush()
        
        try:
            vector_agent.fill()
            sys.stdout.write("Vector store fill successful\n")
        except Exception as e:
            pytest.fail(f"Vector store fill failed: {str(e)}")
        sys.stdout.flush()

    def test_vector_store_delete(self, vector_agent):
        sys.stdout.write("\n=== Testing vector store delete ===\n")
        sys.stdout.flush()
        
        try:
            vector_agent.delete(folders=["test_data/sub1"])
            sys.stdout.write("Vector store delete successful\n")
        except Exception as e:
            pytest.fail(f"Vector store delete failed: {str(e)}")
        sys.stdout.flush()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])