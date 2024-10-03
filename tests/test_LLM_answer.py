from src.generation_utils import LLM_answer,LLM_answer_v2   
import yaml


if __name__=='__main__': 
    
    #load the config/config.yaml file
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test the LLM_answer function
    prompt_1='Qui est Alice ?'
    answer1=LLM_answer(
    prompt_1,
    json_formatting=False,
    pydantic_object=None,
    model_name=config['model_name'],
    temperature=1,
    stream=False,
    llm_provider=config['llm_provider'],
    )
    
    print(answer1)
    
    #Test the Answer in json format
  
    #we create a answer pydantic object
    from pydantic import BaseModel
    class N(BaseModel):
        answer:str
    
    prompt_2='Qui est Alice ?'
    answer2=LLM_answer_v2(
    prompt_2,
    json_formatting=True,
    pydantic_object=N,
    model_name=config['model_name'],
    temperature=1,
    stream=False,
    llm_provider=config['llm_provider'],
    )
    
    
    print("Final Answer:",answer2)
    print("Type of answer2:",type(answer2))
    
    