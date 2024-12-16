from gliner import GLiNER

import json
from src.knowledge_graph import KnowledgeGraphIndex,EntityExtractor

text = """
Elles Bougent Pour l’Orientation MARTZLOFF Alice SCHALCK Elsa 7 décembre 2023 ‹#› Plan 01 02 Témoignage Temps d’échanges ‹#› Alice MARTZLOFF INTERVENANTE Data Scientist à Euranova - Consultante pour un client dans le bâtiment et les énergies - Responsable du projet Sustainable Computing Formation scientifique Bac Scientifique option Maths / Physique Classe préparatoire aux grandes Écoles en PCSI École d’ingénieur.e.s Centrale Marseille et Master d’Intelligence Artificielle et Apprentissage Automatique Personne d’autre que vous n’a le droit de vous dire que quoi vous êtes capable ! ; ; Avoir un impact positif Elsa SCHALCK INTERVENANTE Data Scientist à Euranova Responsable des projets de R&D en imagerie médicale Consultante pour un client dans le microbiote et l’oncologie Formation scientifique Bac Scientifique option Maths Classe préparatoire aux grandes Écoles en BCPST École d’ingénieur.e.s AgroParisTech (Institut national des sciences et industries du vivant et de l'environnement) : Option Ingénierie et santé Spécialité Data Science Comment ai-je été attiré par ces études ? La biologie et les maths.
"""

# labels = ["person", "project", "location", "company"]

# entities = model.predict_entities(text, labels)

# for entity in entities:
#     print(entity["text"], "=>", entity["label"])
    
# list_entities = [entity["text"] for entity in entities]
    


# model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5",map_location='cuda')
# relation='works with'
# relation='works for'
# labels = [k +' <> '+ relation for k in list_entities][0:4]
# print("LABELS:",labels)
# entities = model.predict_entities(text, labels)

# for entity in entities:
#     print(entity["label"], "=>", entity["text"])
    

# Example usage
entity_extractor = EntityExtractor(
    entity_model_name="urchade/gliner_multi-v2.1",
    relation_model_name="knowledgator/gliner-multitask-large-v0.5",
    allowed_entities_path="src/allowed_entities.json",
    allowed_relations_path="src/allowed_relations.json",
    allowed_precise_relations_path="src/allowed_detailled_relations.json"
)



entities = entity_extractor.extract_entities(text)
relations = entity_extractor.extract_precise_relations(text, entities)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
    
print("###########################################################")

for relation in relations:
    print(relation)