import unittest
from main_utils.utils import detect_language  # Replace 'your_module' with the actual module name where detect_language is defined

class TestLanguageDetection(unittest.TestCase):

    def test_english_detection(self):
        text = "languages are awesome"
        result = detect_language(text)
        self.assertEqual(result, 'en', "Should be 'en' for English text")

    def test_french_detection(self):
        text = "les langues sont géniales"
        result = detect_language(text)
        self.assertEqual(result, 'fr', "Should be 'fr' for French text")

if __name__ == '__main__':
    unittest.main()