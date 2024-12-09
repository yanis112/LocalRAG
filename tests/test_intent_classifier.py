from src.aux_utils.text_classification_utils import IntentClassifier
import pytest
import yaml

@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)
    
def 