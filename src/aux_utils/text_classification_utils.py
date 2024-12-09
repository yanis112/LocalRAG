
from functools import lru_cache
import torch
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

class IntentClassifier:
    """
    A classifier for determining the intent of a given text using zero-shot classification.
    Attributes:
        config (dict): The configuration dictionary containing the labels and model details.
        labels (list): The list of class labels for classification.
        pipeline (Pipeline): The zero-shot classification pipeline.
        
       
    Methods:
        __init__(labels):
            Initializes the IntentClassifier with the given labels.
        _get_pipeline():
            Returns a zero-shot classification pipeline with the specified model and device.
        classify(text):
            Classifies the given text and returns the most probable class label.
    """
    def __init__(self, config,labels_dict=None):
        #get the labels dictionary from the config file
        if labels_dict is None:
            self.labels_dict = config["actions_dict"]
            self.labels = list(self.labels_dict.keys())
        else:
            self.labels_dict = labels_dict
            self.labels = list(labels_dict.keys())
        self.pipeline = self._get_pipeline()
        #load the classification prompt template from a txt file
        with open("prompts/llm_text_classification.txt", "r",encoding='utf-8') as file:
            self.classification_prompt = file.read()
        
        # Instantiation using from_template (recommended)
        self.classification_prompt = PromptTemplate.from_template(self.classification_prompt)
        self.query_classification_model = config["query_classification_model"]
        self.query_classification_provider = config["query_classification_provider"]
        
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

    def classify(self, text,method="zero-shot"):
        """
        Classify the given text into one of the predefined labels.

        Args:
            text (str): The text to be classified.

        Returns:
            str or None: The most probable class label if classification is successful, 
                         otherwise None.
        """
        
        if method=="zero-shot":
            results = self.pipeline(text, self.labels)
            if results:
                most_probable_class = max(results["labels"], key=lambda x: results["scores"][results["labels"].index(x)])
                return most_probable_class
            
        elif method=="LLM":
            from src.main_utils.generation_utils_v2 import LLM_answer_v3
            full_prompt=self.classification_prompt.format(user_query=text,labels_dict=str(self.labels_dict))

            answer=LLM_answer_v3(prompt=full_prompt,model_name=self.query_classification_model,llm_provider=self.query_classification_provider,stream=False)
        
            return answer
            
        
        return None

if __name__ == "__main__":
    import time
    #labels = ["question posée", "recherche d'emploie","recherche utilisant internet"]
    #define label dictionary
    labels_dict = {
        "question posée": "question posée sur un sujet quelconque",
        "recherche d'emploie": "recherche d'emploie, d'offre d'emploie sur un site quelconque",
        "recherche utilisant internet": "recherche d'information sur internet"
    }
    
    classifier = IntentClassifier(labels_dict=labels_dict)
    text = "Check tous les jobs sur Marseille"
    start_time = time.time()
    most_probable_class = classifier.classify(text,method="LLM")
    print("Most probable class:", most_probable_class)
    end_time = time.time()
    print("Time taken:", end_time - start_time)
    text2= "Cherche sur internet qu'est ce que le continent de Mu ?"
    most_probable_class2 = classifier.classify(text2,method="LLM")
    print("Most probable class:", most_probable_class2)
    text3="Qui est Yanis ?"
    most_probable_class3 = classifier.classify(text3,method="LLM")
    print("Most probable class:", most_probable_class3)
   