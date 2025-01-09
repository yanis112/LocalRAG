from gliner import GLiNER
import json
from src.knowledge_graph import KnowledgeGraphIndex,EntityExtractor

text = """
Hello, I am a software engineer at Google. I am currently working on a project with Facebook. 
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