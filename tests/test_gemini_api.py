import sys
import pytest
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from unittest.mock import patch

class TestGoogleAI:
    @pytest.fixture(autouse=True)
    def setup(self):
        sys.stdout.write('\n=== Test Setup Starting ===\n')
        sys.stdout.flush()
        
        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=1,
            max_tokens=None,
            timeout=None,
            max_retries=1
        )
        
        sys.stdout.write('=== Test Setup Complete ===\n\n')
        sys.stdout.flush()

    def test_translation(self):
        sys.stdout.write('=== Translation Test Starting ===\n')
        sys.stdout.flush()

        messages = [
            ("system", "You are a helpful assistant that translates English to French."),
            ("human", "I love programming.")
        ]
        
        result = self.llm.invoke(messages)
        assert result is not None
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        
        sys.stdout.write(f'Translation result: {result.content}\n')
        sys.stdout.write('=== Translation Test Complete ===\n')
        sys.stdout.flush()

    

if __name__ == '__main__':
    pytest.main(['-v', __file__])