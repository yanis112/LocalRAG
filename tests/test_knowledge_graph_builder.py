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
    
    print(builder.retrieve_linked_entities("Nexter Robotics"))
       
    
    exit()

    text = """ le soucis c'est que pour les personnes dont le uid n'a pas été défini de manière centralisée,
    il y a un uid sur la machine qui ne correspondrait pas à ce qu'on pourrait forcer depuis jumpcloud. tout a été fait de puis odin et c'est d'ailleur freyja
    l'hôte de slurm et non odin depuis quelle machine est-ce toutes ces commandes sont lancées ? j'ai un souvenir d'un update qui avait été fait par mathias et
    thomas sur les uid et gid de chacun : ils avaient harmonisé tous les uid et gid de chacun entre les machines, pour être de type 90xx. par exemple, pour moi 
    : screenshot je ne sais pas à quel point cela un impact sur slurm, mais j'ai remarqué que ton uid / gid n'était pas de type 90xx @hakima.arroubat @yanis.labeyrie 
    @mohamad.ammar @marouane.najime en écho au précédent message de @alice.martzloff, je vous transmets le lien vers notre dossier tutorials  qui contiendra la réponse 
    à beaucoup de questions it, notamment pour se connecter sur les machines. si vous avez des idées pour pérénniser le process, plutôt que de passer par newgrp à chaque 
    fois ainsi que pour le problème d'affectation à de mauvais groupes, je suis preneur. bonjour à @all , nous nous sommes penché sur slurm avec @alice.martzloff cette
    semaine que nous avons réussi à refaire fonctionner via une solution de contournement. nous allons donc essayer de retourner dessus prochainement. le problème vient 
    d'un problème de group, slurm ayant une taille très limité pour l'affectation aux groupes, ne passant pas l'information de notre appartenance à celui de docker. 
    id lancé en ligne de commande : uid=10010(robin.mancini) gid=1010(docker) groups=1010(docker),27(sudo),33221(nexter),33224(robin.mancini) id lancé dans slurm 
    (qui part en erreur car n'accède pas au group docker) : uid=10010(robin.mancini) gid=33224(robin.mancini) groups=33224(robin.mancini),27(sudo),23445(ai4pet) 
    id lancé via session newgrp docker et qui fonctionne : uid=10010(robin.mancini) gid=1010(docker) groups=1010(docker),27(sudo),23445(ai4pet) autre chose étrange,
    on voit qu'il me met dans le groupe 23445 ai4pet alors que je n'y suis pas : """
    
    
    text2= """Bonjour à tous, je me permets de vous écrire car j'ai un souci avec mon compte jumpcloud et Robin Mancini est le responsable de ce service."""
    
    # Extract relations from text
    builder.complete_from_text(text)
    #builder.complete_from_text(text2)
    
    # Print the knowledge graph
    builder.interactive_graph()
