from src.knowledge_graph import KnowledgeGraphIndex
from src.retrieval_utils import find_loader
import time
import os
import yaml
from tqdm import tqdm
from langchain_core.documents import Document
from main_utils.generation_utils import LLM_answer

if __name__=="__main__":
     # load the document that is in test_docs/test.docx
     #load the config from config/config.yaml
     
     with open("config/config.yaml") as file:
          config = yaml.load(file, Loader=yaml.FullLoader)
     
     
     #find the loader
     folder= 'test_docs/' #'data/mattermost/IT @ FR/'  
     start_time = time.time()
     list_files = os.listdir(folder)                 #"test_docs")
     list_paths = [folder+file for file in list_files]  #"test_docs/"+file 
     list_files.sort()  # Trie les fichiers en ordre alphabétique
     total_chunks=[]
     for k in tqdm(range(len(list_paths))):
          # print("TREATING FILE: ", list_files[k])
          loader = find_loader(full_path=list_paths[k],name=list_files[k])
          text = loader.load()[0].page_content
          #split the text into chunks of 500 words
          list_chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
          #reconstruct the langchain document objects from the chunks
          list_docs=[Document(page_content=chunk) for chunk in list_chunks]
          total_chunks.extend(list_docs)
          
          
     #import the KnowledgeGraphIndex
     kg_index=KnowledgeGraphIndex(config)
     
     db= kg_index.from_documents(documents=total_chunks,overwrite=True)
     
     #get the knowledge graph
     kg = kg_index.get_graph()
     #print all existing nodes$
     print("NODES:",kg.nodes())
     
     
     query = "Qui sont les projets sur lequels travaille Aziz ?"
     found_docs = db.similarity_search(query=query,k=5)
     
     print("FOUND DOCS:",found_docs)
     
     exit()
     
     
     
     
     #get first doc object
     doc = found_docs[0].page_content+str(found_docs[0].metadata)
     
     #create final_prompt using context and query
     final_prompt = "Answer the following question: "+query+" using the following knowledge grap context: "+doc
     
     print("Final Prompt:",final_prompt)
     
     #formulate the answer to the question
     answer=LLM_answer(prompt=final_prompt,model_name='gemma2',llm_provider='ollama',stream=False)
     
     print("ANSWER:",answer)
     
     
     
     #extract subgraph
     # sub=kg_index.extract_subgraph(entities=['BioTech','EURA NOVA'])
     # print("SUBGRAPH:",sub)
    
     
     # #make a cypher query to find the answer
     # cypher_query="""
     # MATCH (A)<-[]-(B)
     # WHERE A.entity_name = "BioTech"
     # RETURN B.entity_name
     # """
     
     
     # #query the graph
     # result = kg_index.query_graph(cypher_query)
     # print("CYPER QUERY RESULT:",result)
     
     