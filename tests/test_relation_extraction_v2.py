from src.knowledge_graph import EntityExtractor



#load config from config/config.yaml
import yaml
with open("config/config.yaml") as file:
    config = yaml.safe_load(file)

entity_extractor = EntityExtractor(config=config)



text = """

Il y a 6 personnes avec des congés restants à prendre avant Juin

Alors c'est une erreur (et c'est même Juin 2022)

same

je dois prendre les congés avant mai 2021 alors que sur Eurécia c'est mai 2022. c'est pour l'année N

Quel pb ?

Yes (après, attend pas non plus une réactivité à l'heure :))

Je risque d'avoir quelques MR à faire relire, je passe plutôt par toi que par Antoine ?

Hello @all, communication générale. Antoine va travailler sur une biblio un peu expresse pour Tchek dans les 3 prochaines semaines. Merci de ne pas le solliciter (excepté sur Crocos et Heimdallr qui sont prévus dans son EDT). Si vous avez des questions techniques, essayez de vous tourner vers quelqu'un d'autre en priorité. Perso, je peux me dégager du temps plus facilement que d'habitude ce mois-ci

bulletin de salaire reçu! merci
Il y a juste un problème de date d'expiration pour l'année n-1
"""

#use the entity extractor to extract entities and relations

import time
#measure execution time
start_time = time.time()
entities=entity_extractor.extract_entities(text)
relations_1=entity_extractor.extract_relations_v3(text) #,entities)
relations_2=entity_extractor.extract_precise_relations(text,entities)
end_time = time.time()

print("Execution time:",end_time-start_time)
print("RELATIONS:",relations_1)
print("PRECISE RELATIONS:",relations_2)