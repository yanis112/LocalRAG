


from src.retrieval_utils import query_database_v2
from main_utils.generation_utils import RAG_answer, pull_model


if __name__=='__main__':
    
    # #load the config file
    # import yaml
    # with open('config/config.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    
    # # Test the query_database_v2 function
    # query = "Qui est Alice ?"
    # output=query_database_v2(query,default_config=config,config={})
    
    # print(output)
    # print("###########################################")
    
    # #test the rag answer
    # output_rag=RAG_answer(query,default_config=config,config={'return_chunks':True})
    
    # print(output_rag)
    
    print(pull_model("llama3"))