import argparse
import os
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time

"""PAY ATTENTION, THIS SCRIPT CANNOT BE RUN IN A DOCKER CONTAINER ! LAUNCH IT OUTSIDE THE CONTAINER, THIS SCRIPT DOESNT WORK YET"""

def save_cookies(driver, path):
    """
    Sauvegarde les cookies de la session du driver.

    Parameters:
    - driver: Objet WebDriver
    - path: Chemin pour sauvegarder les cookies

    Returns:
    None
    """
    cookies = driver.get_cookies()
    with open(path, "wb") as file:
        pickle.dump(cookies, file)
    print(f"Cookies sauvegardés dans {path}. Vous pouvez maintenant les utiliser pour les futures sessions.")

def main():
    parser = argparse.ArgumentParser(description="Obtenir des cookies via une connexion manuelle.")
    parser.add_argument(
        "--directory",
        type=str,
        default="data/pages",
        help="Répertoire pour sauvegarder les cookies (par défaut : data/pages)"
    )
    args = parser.parse_args()
    directory = args.directory

    # Créer le répertoire s'il n'existe pas
    os.makedirs(directory, exist_ok=True)

    chrome_options = Options()
    # Retirer les options qui peuvent empêcher l'affichage de la fenêtre
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--disable-gpu")
    
    # Ajouter des options pour une meilleure visibilité
    chrome_options.add_argument("--window-size=1920,1200")
    chrome_options.add_argument("--start-maximized")
    
    # Assurer que le mode headless n'est pas activé
    chrome_options.headless = False

    try:
        # Utiliser webdriver-manager pour gérer ChromeDriver
        #service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(options=chrome_options) #service=service,
        driver.get("https://app.happeo.com/home")
        time.sleep(5)
        print("Veuillez vous connecter manuellement dans la fenêtre du navigateur ouverte.")
        input("Appuyez sur Entrée après avoir terminé la connexion...")

        save_cookies(driver, os.path.join(directory, "cookies_happeo.pkl"))

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
    finally:
        driver.quit()
        print("Navigateur fermé.")

if __name__ == "__main__":
    main()
