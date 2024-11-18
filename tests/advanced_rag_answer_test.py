from src.main_utils.generation_utils import advanced_RAG_answer
import yaml




def test_advanced_RAG_answer():
    #load the config file config/config.yaml
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    query = "In which place did the person that works on AI4PET did study ?"
    query2 = "Quels sont les machines du compute lap avec une VRAM supérieure à 16GB ?"
    query3 = "Quels sont les personnes qui ont travaillé sur des données temporelles et savent utiliser des réseaux de neurones LSTM ?"
    query4= "Ou a étudié le stagiaire qui travaille avec Rosana ?"
    answer,sources = advanced_RAG_answer(query4, default_config=config)
    print("##############################################")
    print("Answer : ")
    return answer
    
if __name__ == "__main__":
    print(test_advanced_RAG_answer())
    