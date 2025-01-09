import pytest
import sys
import time
from pathlib import Path
from src.aux_utils.vision_utils_v2 import ImageAnalyzerAgent

TEST_IMAGE = "aux_data/document_scores.png"
TEST_PROMPT = "Describe precisely the content of the image."

class TestImageAnalyzer:
    @pytest.fixture(scope="class")
    def analyzer(self):
        return ImageAnalyzerAgent()

    @pytest.fixture(scope="class")
    def image_path(self):
        path = Path(TEST_IMAGE)
        assert path.exists(), f"Test image not found: {TEST_IMAGE}"
        return str(path)

    def _run_provider_test(self, analyzer, image_path, provider, model):
        sys.stdout.write(f"\n=== Testing {provider} Provider ===\n")
        sys.stdout.flush()

        start_time = time.time()
        description = analyzer.describe(
            image_path, 
            prompt=TEST_PROMPT,
            vllm_provider=provider,
            vllm_name=model
        )
        duration = time.time() - start_time

        sys.stdout.write(f"Description length: {len(description)}\n")
        sys.stdout.write(f"Processing time: {duration:.2f} seconds\n")
        sys.stdout.write("=== Test Complete ===\n")
        sys.stdout.flush()

        assert description, f"{provider} description should not be empty"
        assert duration < 30, f"{provider} processing took too long"
        return description, duration

    def test_groq_description(self, analyzer, image_path):
        description, _ = self._run_provider_test(
            analyzer, image_path, "groq", "llama-3.2-90b-vision-preview"
        )
        assert isinstance(description, str)
        assert len(description) > 50

    def test_gemini_description(self, analyzer, image_path):
        description, _ = self._run_provider_test(
            analyzer, image_path, "gemini", "gemini-1.5-flash"
        )
        assert isinstance(description, str)
        assert len(description) > 50

    def test_github_description(self, analyzer, image_path):
        description, _ = self._run_provider_test(
            analyzer, image_path, "github", "gpt-4o-mini"
        )
        assert isinstance(description, str)
        assert len(description) > 50

    def test_invalid_provider(self, analyzer, image_path):
        with pytest.raises(ValueError):
            analyzer.describe(
                image_path,
                prompt=TEST_PROMPT,
                vllm_provider="invalid",
                vllm_name="invalid-model"
            )