import json
import ollama

def triplextract(model, text, entity_types, predicates):
    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
      
        **Entity Types:**
        {entity_types}
        
        **Predicates:**
        {predicates}
        
        **Text:**
        {text}
        """

    message = input_format.format(
                entity_types=json.dumps({"entity_types": entity_types}),
                predicates=json.dumps({"predicates": predicates}),
                text=text)

    messages = [{'role': 'user', 'content': message}]
    options={"temperature": 0.0}
    response = ollama.chat(model=model, messages=messages, options=options)
    output = response['message']['content']
    return output

model = 'sciphi/triplex'

entity_types = ["PERSON", "COMPANY", "PROJECT"]
predicates = ["WORKS_FOR", "WORKS_WITH", "CONTRIBUTES_TO"]
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

def parse_output_to_triples(output):
    try:
        # Remove leading and trailing non-JSON parts
        json_str_start = output.find('{')
        json_str_end = output.rfind('}') + 1
        clean_json_str = output[json_str_start:json_str_end]
        
        data = json.loads(clean_json_str)

        triples_list = data['entities_and_triples']
        entities = {}
        triples = []

        # Process entities
        for item in triples_list:
            if ',' in item and ':' in item:  # Likely an entity
                parts = item.split(',', 1)
                entity_id = parts[0].strip("[] ")
                entity_type, entity_name = parts[1].split(':', 1)
                entities[entity_id] = {'type': entity_type.strip(), 'name': entity_name.strip()}

        # Process relationships
        for item in triples_list:
            if item.count('[') == 2 and item.count(']') == 2 and ' ' in item:  # Likely a relationship
                parts = item.split(' ')
                if len(parts) == 3:
                    entity1_id = parts[0].strip("[]")
                    relation = parts[1].strip()
                    entity2_id = parts[2].strip("[]")
                    if entity1_id in entities and entity2_id in entities:
                        entity1_name = entities[entity1_id]['name']
                        entity2_name = entities[entity2_id]['name']
                        triples.append((entity1_name, relation, entity2_name))

        return triples
    except json.JSONDecodeError:
        return []

prediction = triplextract(model, text, entity_types, predicates)
print("#######################################")
print(prediction)

print("#######################################")
parsed_triples = parse_output_to_triples(prediction)
print(parsed_triples)