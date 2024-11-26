
from functools import lru_cache
from transformers import pipeline
import torch

class IntentClassifier:
    """
    A classifier for determining the intent of a given text using zero-shot classification.
    Attributes:
        labels (list): A list of possible labels for classification.
        pipeline (Pipeline): A Hugging Face pipeline for zero-shot classification.
    Methods:
        __init__(labels):
            Initializes the IntentClassifier with the given labels.
        _get_pipeline():
            Returns a zero-shot classification pipeline with the specified model and device.
        classify(text):
            Classifies the given text and returns the most probable class label.
    """
    def __init__(self, labels):
        self.labels = labels
        self.pipeline = self._get_pipeline()

    @lru_cache(maxsize=None)
    def _get_pipeline(self):
        """
        Retrieves a zero-shot classification pipeline with caching.

        This method uses the Hugging Face `pipeline` function to create a zero-shot
        classification pipeline with the specified model and device. The result is
        cached to improve performance on subsequent calls.

        Returns:
            Pipeline: A zero-shot classification pipeline configured with the 
            "knowledgator/comprehend_it-base" model, running on a CUDA device with 
            torch float16 precision.
        """
        return pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base", device='cuda',torch_dtype=torch.float16)

    def classify(self, text):
        """
        Classify the given text into one of the predefined labels.

        Args:
            text (str): The text to be classified.

        Returns:
            str or None: The most probable class label if classification is successful, 
                         otherwise None.
        """
        results = self.pipeline(text, self.labels)
        if results:
            most_probable_class = max(results["labels"], key=lambda x: results["scores"][results["labels"].index(x)])
            return most_probable_class
        return None

if __name__ == "__main__":
    import time
    labels = ["question posée", "recherche d'emploie","recherche utilisant internet"]
    classifier = IntentClassifier(labels)
    text = "Check tous les jobs sur Marseille"
    start_time = time.time()
    most_probable_class = classifier.classify(text)
    print("Most probable class:", most_probable_class)
    end_time = time.time()
    print("Time taken:", end_time - start_time)
    text2= "Cherche sur internet qu'est ce que le continent de Mu ?"
    most_probable_class2 = classifier.classify(text2)
    print("Most probable class:", most_probable_class2)
    text3="Qui est Yanis ?"
    most_probable_class3 = classifier.classify(text3)
    print("Most probable class:", most_probable_class3)
   