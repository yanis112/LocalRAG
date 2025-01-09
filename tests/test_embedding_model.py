import sys
import pytest
from src.main_utils.embedding_utils import get_embedding_model

class TestEmbeddingModel:
    """Test suite for embedding model functionality"""
    
    def test_model_loading_and_inference(self):
        """Test embedding model loading and basic inference"""
        sys.stdout.write('\n=== Testing Model Loading ===\n')
        sys.stdout.flush()
        
        # Test model loading
        model = get_embedding_model("jinaai/jina-embeddings-v3", show_progress=True)
        assert model is not None, "Model failed to load"
        
        sys.stdout.write('\n=== Testing Inference ===\n')
        sys.stdout.flush()
        
        # Test inference
        test_text = "Ceci est un document de test"
        embedding = model.embed_documents(test_text)
        
        # Validate embedding
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        
        sys.stdout.write(f'\nModel loaded successfully\n')
        sys.stdout.write(f'Embedding length: {len(embedding)}\n')
        sys.stdout.write('=== Test Complete ===\n')
        sys.stdout.flush()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])