from src.main_utils.generation_utils_v2 import LLM_answer_v3
from langchain.tools import tool
from src.aux_utils.job_scrapper_v2 import JobAgent


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

print(multiply.name)
print(multiply.description)
print(multiply.args)

if __name__=="__main__":
    prompt="multiply those two numbers: {'first_int': 4, 'second_int': 5} and also say 'bonjour'"
    #prompt="Trouve des offres d'emplois de datascientist sur aix en provence"
    answer,tool_calls=LLM_answer_v3(
        prompt=prompt,
        model_name= "gemini-2.0-flash", #"llama-3.3-70b-versatile",
        llm_provider="google", #"ollama", #,
        temperature=1,
        stream=False,
        tool_list=[multiply],
        
    )
    print("Answer:",answer)
    print("##################################################")
    print("Tool calls:",tool_calls)
    
    