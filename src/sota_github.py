import telebot
import requests
import time
from src.vision_utils import UniversalImageLoader

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
def handle_image(message):
    try:
        # Télécharger l'image
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Sauvegarder l'image localement
        image_path = "received_image.png"
        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        
        # Traiter l'image et mesurer le temps pris
        start_time = time.time()
        loader = UniversalImageLoader()
        res = loader.universal_extract(image_path)
        end_time = time.time()
        
        # Calculer le temps pris
        time_taken = end_time - start_time
        
        # Envoyer la réponse à l'utilisateur
        bot.reply_to(message, f"Résultat du traitement : {res}\nTemps pris : {time_taken:.2f} secondes")
    except Exception as e:
        bot.reply_to(message, f"Une erreur s'est produite : {str(e)}")

# Démarrage du bot
if __name__ == '__main__':
    print("Le bot est en cours d'exécution...")
    bot.polling()