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
    with open(cv_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_user_info(info_path):
    with open(info_path, 'r') as file:
        return yaml.safe_load(file)

def create_prompt(message_type, user_message, include_cv=False, cv_path=None):
    if message_type not in EXAMPLE_MESSAGES:
        raise ValueError(f"Type de message '{message_type}' non supporté.")
    
    template = EXAMPLE_MESSAGES[message_type]
    
    print("Type of template:", type(template))
    print("Type of user_message:", type(user_message))
    
    
    prompt="### Instructions:\n"+user_message+"\n\n### Tu t'inspirera du template et du style de message suivant pour rédiger le contenu demandé:\n\n"+template
    
    # Remplacer les placeholders dans le template si nécessaire
   
    if include_cv and cv_path:
        cv_content = read_cv(cv_path)
        prompt += "\n\n## Rédigez la lettre en mettant en relation les compétences utiles de mon CV pour cette offre (pas toutes uniquement une ou deux, les plus pertinantes vis-à-vis de l'offre):\n\n"
        prompt += cv_content
    
    #add the usefull information to the prompt
    info_dict=read_user_info("info.yaml")
    prompt += "\n\n### Informations utiles:\n" + "Nom: " + info_dict["nom_complet"] + " Téléphone: " + info_dict["numero_telephone"]

    return prompt

def auto_job_writter(user_input, info_path, cv_path):
    # Classify the user's intention
    message_type = classify_intention(user_input)
    print("Intention classifiée:", message_type)
    
    # Read user information from the YAML file
    user_info = read_user_info(info_path)
    
    # Create the prompt
    prompt = create_prompt(message_type, user_input, include_cv=True, cv_path=cv_path)
    
    return prompt

# Example usage
if __name__ == "__main__":
    user_input = "Je voudrais postuler pour un poste de Data Scientist chez votre entreprise."
    info_path = "info.yaml"
    cv_path = "cv.txt"
    prompt = auto_job_writter(user_input, info_path, cv_path)
    print(prompt)