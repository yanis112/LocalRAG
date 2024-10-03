import json



def test2():

    # Charger le fichier JSON (remplacez 'your_file.json' par le chemin de votre fichier)
    with open('data_archive/story.json', 'r') as file:
        data = json.load(file)

    # Extraire les textes des assets de type 'StoryAsset'
    story_texts = [asset['story']['id'] for asset in data['node']['assets'] if asset['__typename'] == 'StoryAsset']

    # Fonction pour parcourir récursivement les nodes et extraire le texte
    def extract_text_from_nodes(nodes):
        texts = []
        for node in nodes:
            if node['object'] == 'text':
                texts.append(node['text'])
            elif 'nodes' in node:
                texts.extend(extract_text_from_nodes(node['nodes']))
        return texts

    # Extraire le texte du corps principal
    body_texts = []
    if 'body' in data['node'] and 'state' in data['node']['body'] and 'document' in data['node']['body']['state']:
        body_texts = extract_text_from_nodes(data['node']['body']['state']['document']['nodes'])

    # Combinaison de tous les textes extraits
    all_texts = story_texts + body_texts

    #c'est une liste de mot body_texts, regroupe les pour avoir un seul texte
    body_texts = ' '.join(body_texts)

    print("Story texts:", story_texts)
    print("######################################################")
    print("Body texts:", body_texts)

import json

def load_document_content_and_metadata(json_path):
    """
    Load the content and extract important metadata from a JSON document.

    Parameters:
    json_path (str): The file path of the JSON document.

    Returns:
    tuple: A tuple containing:
        - str: The concatenated text content of the document, including hyperlinks and paths.
        - dict: Important metadata including title, author, date, and uuid.
    """
    # Load the JSON data from the specified file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Access the main body of the document
    nodes = data['node']['body']['state']['document']['nodes']

    # Extract and concatenate the text content from the nodes
    text_content = ""
    for node in nodes:
        for subnode in node.get('nodes', []):
            text = subnode.get('text', '')
            if 'url' in subnode.get('data', {}):  # Check if there is a hyperlink
                text = f"{text} ({subnode['data']['url']})"
            text_content += text + " "

    # Extract important metadata
    metadata = {
        'title': data['node']['title'],
        'author': data['node']['author']['id'],
        'date': data['node']['created'],
        'uuid': data['node']['uuid']
    }

    return text_content, metadata

if __name__ == "__main__":
    content,metadata=load_document_content_and_metadata('data_archive/test2.json')
    print("CONTENT:",content)
    print("#################################")
    print("METADATA:",metadata)
    print("#################################")