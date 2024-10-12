import functools
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

class IntentClassifier:
    def __init__(self, labels):
        self.labels = labels
        self.pipeline = self._get_pipeline()

    @functools.lru_cache(maxsize=1)
    def _get_pipeline(self):
        model = GLiClassModel.from_pretrained("knowledgator/gliclass-large-v1.0-init")
        tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-large-v1.0-init")
        return ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')

    def classify(self, text):
        results = self.pipeline(text, self.labels, threshold=0.5)[0]  # because we have one text
        if results:
            most_probable_class = max(results, key=lambda x: x["score"])["label"]
            return most_probable_class
        return None

if __name__ == "__main__":
    labels = ["demande de photo","demande de conversation"]
    classifier = IntentClassifier(labels)
    text = "Salut ! je peux voir tes pieds ?"
    most_probable_class = classifier.classify(text)
    
    print("Most probable class:", most_probable_class)