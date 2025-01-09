import sys
import pytest
from typing import Optional
from src.main_utils.utils import detect_language

class TestLanguageDetection:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        sys.stdout.write('\n=== Starting Language Detection Test ===\n')
        sys.stdout.flush()
        yield
        sys.stdout.write('=== Test Complete ===\n')
        sys.stdout.flush()

    def _run_detection_test(self, text: str, expected: str) -> None:
        try:
            sys.stdout.write(f'\nTesting text: "{text}"\n')
            result = detect_language(text)
            sys.stdout.write(f'Detected language: {result}\n')
            assert result == expected
            sys.stdout.write('✓ Test passed\n')
        except Exception as e:
            sys.stdout.write(f'✗ Test failed: {str(e)}\n')
            raise
        finally:
            sys.stdout.flush()

    def test_english_detection(self):
        self._run_detection_test(
            "The quick brown fox jumps over the lazy dog",
            "en"
        )

    def test_french_detection(self):
        self._run_detection_test(
            "Les langues sont géniales",
            "fr"
        )

  