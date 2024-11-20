import time

# Dictionary to store import times
import_times = {}

# Measure time to import telebot
start_time = time.time()
import telebot
end_time = time.time()
import_times['telebot'] = end_time - start_time

# Measure time to import requests
start_time = time.time()
import requests
end_time = time.time()
import_times['requests'] = end_time - start_time

# Measure time to import ImageAnalyzer
start_time = time.time()
from src.image_analysis import ImageAnalyzer
end_time = time.time()
import_times['ImageAnalyzer'] = end_time - start_time

# Print the time taken to import each module
for module, duration in import_times.items():
    print(f"Time taken to import {module}: {duration:.6f} seconds")

# Remplacez 'YOUR_BOT_TOKEN' par le token obtenu via BotFather
TOKEN = '7871847224:AAGh9s0DFUjT-YQW0L6nA6L2HRHpJ69p99c'

# Création de l'instance du bot
bot = telebot.TeleBot(TOKEN)

# Gestionnaire pour la commande /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Bonjour ! Je suis votre nouveau bot Telegram. Comment puis-je vous aider ?")

# Gestionnaire pour les messages texte
@bot.message_handler(func=lambda message: message.text is not None)
def echo_all(message):
    bot.reply_to(message, f"Vous avez dit : {message.text}")

# Gestionnaire pour les messages avec des images
@bot.message_handler(content_types=['photo'])
def handle_images(message):
    try:
        # We get the doc from github
        from github import Github
        import os
        g = Github(os.getenv("GITHUB_TOKEN"))
        repo = g.get_repo("yanis112/SOTA_machine_learning")
        # Get the content of the README.md file
        file = repo.get_contents("README.md")
        current_content = file.decoded_content.decode()
        
        # We find the section that is the good one
        from src.long_doc_editor import LongDocEditor
        editor = LongDocEditor(current_content)
        list_sections = editor.get_sections()
        editor.compute_embeddings()

        for i, photo in enumerate(message.photo):
            # Télécharger l'image
            file_info = bot.get_file(photo.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            
            # Sauvegarder l'image localement avec un nom unique
            image_path = f"received_image_{i}.png"
            with open(image_path, 'wb') as new_file:
                new_file.write(downloaded_file)
            
            # We structure-ocerise the image to obtain the text
            analyser = ImageAnalyzer(model_name="gpt-4o-mini")
            prompt = """Extract the technical information from the image. You will only extract the info explaining why is good (better than current SOTA), and for which task (computer vision / natural language processing , ect... ), and maybe a link, nothing else. You will format it in markdown in the following way, EXEMPLE: * **BifRefNet**: A State-of-the-Art Background Removal Model  
    + [BifRefNet](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo) 🕊️ (free) is a highly performant background removal model that achieves high accuracy on various images. """
            description = analyser.describe_advanced(image_path, prompt=prompt, grid_size=1)
            print("### Description found:", description)
            
            most_relevant = editor.most_relevant_section(description)
            print("### Most relevant section found:", most_relevant)
            
            # Insert the generated description at the end of the most relevant section
            end_line = most_relevant['end_line']
            lines = current_content.split('\n')
            lines.insert(end_line + 1, description)
            current_content = '\n'.join(lines)
        
        # Update the README.md file in the repository
        repo.update_file(file.path, "Update README with new sections", current_content, file.sha)
        
        # Envoyer la réponse à l'utilisateur
        bot.reply_to(message, "Les images ont été traitées et le README a été mis à jour.")
    except Exception as e:
        bot.reply_to(message, f"Une erreur s'est produite : {str(e)}")

# Démarrage du bot
if __name__ == '__main__':
    print("Le bot est en cours d'exécution...")
    bot.polling()