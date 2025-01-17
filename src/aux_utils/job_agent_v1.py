import yaml
from transformers import pipeline

# Load the zero-shot classification model
classifier = pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base",device=0)

# Example message templates
EXAMPLE_MESSAGES = {
    "candidature_spontanee": """
Objet : Candidature spontanée pour un poste de Data Scientist / Machine Learning Engineer chez {nom_entreprise}

Madame, Monsieur,

Passionné par l'analyse de données et l'intelligence artificielle, je vous adresse ma candidature spontanée pour un poste de Data Scientist, Consultant Big Data, ou Machine Learning Engineer au sein de {nom_entreprise}.

[Votre message personnalisé ici]

Cordialement,
{nom_complet}
Contact: {numero_telephone}
""",
    "relance": """
Objet : Relance de ma candidature pour le poste de {nom_poste} chez {nom_entreprise}

Madame, Monsieur,

Je me permets de vous relancer concernant ma candidature pour le poste de {nom_poste} au sein de {nom_entreprise}. Je suis très motivé par cette opportunité et je souhaite réitérer mon intérêt pour ce poste.

[Votre message personnalisé ici]

Cordialement,
{nom_complet}
Contact: {numero_telephone}
""",
    "first contact": """
Objet : Prise de contact pour une opportunité de collaboration

Madame, Monsieur,

Je me permets de vous contacter pour discuter d'une opportunité de collaboration au sein de {nom_entreprise}. Je suis très intéressé par les projets que vous menez et je souhaiterais en savoir plus sur les possibilités de collaboration.

[Votre message personnalisé ici]

Cordialement,
{nom_complet}
Contact: {numero_telephone}
""",
    "contact_linkedin": """
Objet : Prise de contact via LinkedIn

Madame, Monsieur,

Je me permets de vous contacter via LinkedIn pour discuter d'une opportunité de collaboration au sein de {nom_entreprise}. Je suis très intéressé par les projets que vous menez et je souhaiterais en savoir plus sur les possibilités de collaboration.

[Votre message personnalisé ici]

Cordialement,
{nom_complet}
Contact: {numero_telephone}
"""
}

def classify_intention(user_input):
    candidate_labels = ["candidature_spontanee", "relance", "first contact", "contact_linkedin"]
    result = classifier(user_input, candidate_labels)
    return result['labels'][0]

def read_cv(cv_path):
    """
    Read the content of a file at the given path.

    Args:
        cv_path (str): The path to the file to be read.

    Returns:
        str: The content of the file as a string.
    """
    with open(cv_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_user_info(info_path):
    """
    Read user information from a YAML file.

    Args:
        info_path (str): The path to the YAML file containing user information.

    Returns:
        dict: A dictionary containing the user information loaded from the YAML file.
    """
    with open(info_path, 'r') as file:
        return yaml.safe_load(file)

def create_prompt(message_type, user_message, include_cv=False, cv_path=None):
    """
    Generate a prompt based on the given message type and user message.
    Args:
        message_type (str): The type of message to generate. Must be a key in EXAMPLE_MESSAGES.
        user_message (str): The user's message to include in the prompt.
        include_cv (bool, optional): Whether to include CV content in the prompt. Defaults to False.
        cv_path (str, optional): The file path to the CV. Required if include_cv is True. Defaults to None.
    Returns:
        str: The generated prompt.
    Raises:
        ValueError: If the message_type is not supported.
    """
    if message_type not in EXAMPLE_MESSAGES:
        raise ValueError(f"Type de message '{message_type}' non supporté.")
    
    template = EXAMPLE_MESSAGES[message_type]
    
    prompt="### Instructions:\n"+user_message+"\n\n### Tu t'inspirera du template et du style de message suivant pour rédiger le contenu demandé:\n\n"+template
    
    # Remplacer les placeholders dans le template si nécessaire
   
    if include_cv and cv_path:
        cv_content = read_cv(cv_path)
        prompt += "\n\n## Rédigez la lettre en mettant en relation les compétences utiles de mon CV pour cette offre (pas toutes uniquement une ou deux, les plus pertinantes vis-à-vis de l'offre):\n\n"
        prompt += cv_content
    
    #add the usefull information to the prompt
    info_dict=read_user_info("aux_data/info.yaml")
    prompt += "\n\n### Informations utiles:\n" + "Nom: " + info_dict["nom_complet"] + " Téléphone: " + info_dict["numero_telephone"]

    return prompt

def auto_job_writter(user_input, info_path, cv_path):
    """
    Generate a prompt based on the user's input and additional information.
    This function classifies the user's intention from the input, reads user
    information from a YAML file, and creates a prompt that includes the user's
    CV if specified.
    Args:
        user_input (str): The input provided by the user.
        info_path (str): The file path to the YAML file containing user information.
        cv_path (str): The file path to the user's CV.
    Returns:
        str: The generated prompt based on the user's input and additional information.
    """
    # Classify the user's intention
    message_type = classify_intention(user_input)
    print("Intention classifiée:", message_type)
    
    
    # Create the prompt
    prompt = create_prompt(message_type, user_input, include_cv=True, cv_path=cv_path)
    
    return prompt

# Example usage
if __name__ == "__main__":
    user_input = "Je voudrais postuler pour un poste de Data Scientist chez votre entreprise."
    info_path = "aux_data/info.yaml"
    cv_path = "aux_data/cv.txt"
    prompt = auto_job_writter(user_input, info_path, cv_path)
    print(prompt)