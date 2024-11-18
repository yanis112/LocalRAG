import functools
from transformers import pipeline

class IntentClassifier:
    def __init__(self, labels):
        self.labels = labels
        self.pipeline = self._get_pipeline()

    @functools.lru_cache(maxsize=None)
    def _get_pipeline(self):
        return pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base", device='cuda')

    def classify(self, text):
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
   