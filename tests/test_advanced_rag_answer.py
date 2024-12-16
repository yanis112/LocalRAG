import time
import yaml
from src.main_utils.generation_utils_v2 import RAGAgent
 
 
if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # query = "Qui est Simon Boiko ?"
    # agent = RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})
    # answer = agent.RAG_answer(query)
    # print("Answer:", answer)
    # exit()
    #test internet rag
    agent = RAGAgent(default_config=config, config={"stream": False, "return_chunks": False})
    query = "Est il vrai que Elon Musk a proposé de l'argent à des américains pour qu'ils votent pour Trump explicitement ?"
    start_time = time.time()
    stream_generator = agent.advanced_RAG_answer(query)
    end_time = time.time()
    print("FINAL_ANSWER:",stream_generator)