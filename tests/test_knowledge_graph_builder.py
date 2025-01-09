from src.knowledge_graph import KnowledgeGraphBuilder
from src.retrieval_utils import find_loader
import os
import time
from tqdm import tqdm
import yaml

if __name__ == "__main__":
    
    #load the config from config/config.yaml
    with open("config/config.yaml") as file:
        config = yaml.safe_load(file)
        
    # Create a KnowledgeGraphBuilder object
    builder = KnowledgeGraphBuilder(config)
    
    #make a os listdire to find all files in the folder "test_docs"
    #for each file in the folder, load the file and extract the text
    folder= 'data/mattermost/IT @ FR/'   #'data/gitlab/'              #'data/mattermost/ENX @ FR/'
    start_time = time.time()
    list_files = os.listdir(folder)                 #"test_docs")
    list_paths = [folder+file for file in list_files]  #"test_docs/"+file 
    list_files.sort()  # Trie les fichiers en ordre alphabétique
    
    for k in tqdm(range(len(list_paths)),desc="Loading files and extracting relations"):
        # print("TREATING FILE: ", list_files[k])
        loader = find_loader(full_path=list_paths[k],name=list_files[k])
        text = loader.load()[0].page_content
        #split the text into chunks of 500 words
        list_chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
        for k in range(len(list_chunks)):
            builder.complete_from_text_v3(list_chunks[k])
        
    #we launch the disambiguation process
    builder.disambiguate_entities()
        
    end_time = time.time()
    print("Time to complete the graph: ", end_time - start_time)
    
    builder.interactive_graph(communities=True)

    #retrieve all entities linked to els
    # exita
    
    exit()
    
    print(builder.retrieve_linked_entities("Alice"))
       
    
    exit()

    text = """Bonjour à tous, je me permets de vous écrire car j'ai un souci avec mon compte jumpcloud et R. est le responsable de ce service."""
    
    
    text2= """Bonjour à tous, je me permets de vous écrire car j'ai un souci avec mon compte jumpcloud et Robin Mancini est le responsable de ce service."""
    
    # Extract relations from text
    builder.complete_from_text(text)
    #builder.complete_from_text(text2)
    
    # Print the knowledge graph
    builder.interactive_graph()
