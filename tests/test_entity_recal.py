from src.evaluation_utils import entity_recall

import nltk
nltk.download('punkt')

import nltk
from nltk.data import find

# Téléchargez les données nécessaires
nltk.download('punkt')

# Ajoutez le chemin des données NLTK
nltk.data.path.append('/home/user/nltk_data')


# def test_entity_recal():
#     docs=["Nous avons parlé avec Dylan et Bob. Ils sont très sympas.","Qui est Dylan"]
#     ground_truth="Oui je sais qui est Alice. Elle est très sympa, j'ai parlé à Bob aussi."
#     value=entity_recall(ground_truth,docs)
#     print(value)
    
# if __name__ == "__main__":
#     test_entity_recal()
    