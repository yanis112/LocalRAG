import time
import pyautogui
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import platform
import sys

def read_log_file(log_file_path):
    actions = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            timestamp_str, action_description = line.split(' - ', 1)
            timestamp = float(timestamp_str)
            action = {}
            action['timestamp'] = timestamp
            if action_description.startswith('Mouse Position: x='):
                m = re.match(r'Mouse Position: x=(\d+), y=(\d+)', action_description)
                if m:
                    action['type'] = 'mouse_move'
                    action['x'] = int(m.group(1))
                    action['y'] = int(m.group(2))
            elif action_description.startswith('Event Type: '):
                parts = action_description.split(' - ')
                event_type_part = parts[0]
                event_type = event_type_part[len('Event Type: '):]
                action['type'] = event_type.lower()
                for part in parts[1:]:
                    if ': ' in part:
                        key, value = part.split(': ', 1)
                        key = key.strip().lower().replace(' ', '_')
                        if key == 'framepath':
                            action[key] = eval(value.strip())
                        else:
                            action[key] = value.strip()
            else:
                action['type'] = 'unknown'
            actions.append(action)
    return actions

if __name__ == "__main__":
    acceleration_factor = 2  # Ajuster cette valeur pour accélérer ou ralentir les actions

    # Chemin vers le fichier de log
    log_file_path = "actions_log.txt"

    # Lire les actions depuis le fichier de log
    actions = read_log_file(log_file_path)

    # Trier les actions par timestamp
    actions.sort(key=lambda x: x['timestamp'])

    # Calculer le temps écoulé depuis le début pour chaque action
    first_action_timestamp = actions[0]['timestamp']

    for action in actions:
        action['time_since_start'] = (action['timestamp'] - first_action_timestamp + 3.0) / acceleration_factor  # Ajouter un délai initial de 3 secondes

    # Configurer le WebDriver
    driver = webdriver.Firefox(service=FirefoxService())  # Assurez-vous que geckodriver est dans votre PATH
    driver.set_window_position(0, 0)
    driver.maximize_window()

    # Naviguer vers l'URL spécifiée
    url = "https://www.bing.com/images/create"
    driver.get(url)

    # Attendre que la page se charge complètement
    time.sleep(5 / acceleration_factor)  # Ajuster si nécessaire

    # Récupérer le titre actuel de la fenêtre
    current_window_title = driver.title
    print(f"Titre de la fenêtre actuelle : '{current_window_title}'")

    # Mettre la fenêtre du navigateur au premier plan
    if platform.system() == "Windows":
        try:
            import win32gui, win32con
            hwnd = win32gui.FindWindow(None, current_window_title)
            if hwnd:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                print("Fenêtre Firefox mise au premier plan avec succès.")
            else:
                print("Fenêtre Firefox non trouvée. Utilisation de pyautogui pour cliquer.")
                pyautogui.click(100, 100)  # Cliquer pour amener la fenêtre au premier plan
        except Exception as e:
            print(f"Erreur lors de la mise au premier plan de la fenêtre : {e}")
            pyautogui.click(100, 100)
    elif platform.system() == "Darwin":  # macOS
        import os
        os.system("osascript -e 'tell application \"Firefox\" to activate'")
    else:
        # Pour Linux, utiliser pyautogui pour cliquer sur la fenêtre du navigateur
        pyautogui.click(100, 100)

    # Initialiser WebDriverWait
    wait = WebDriverWait(driver, 20)

    # Enregistrer le temps de début
    start_time = time.time()

    # Variable pour suivre la dernière URL
    last_url = driver.current_url

    # Fonction pour se déplacer vers la frame en utilisant framePath
    def switch_to_frame(frame_path):
        driver.switch_to.default_content()
        for idx in frame_path:
            frames = driver.find_elements(By.TAG_NAME, 'iframe') + driver.find_elements(By.TAG_NAME, 'frame')
            if idx < len(frames):
                driver.switch_to.frame(frames[idx])
            else:
                print(f"Index de frame {idx} hors limites.")
                break

    # Exécuter les actions
    for idx, action in enumerate(actions):
        current_time = time.time()
        time_to_wait = action['time_since_start'] - (current_time - start_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # Exécuter l'action
        if action['type'] == 'mouse_move':
            x = action['x']
            y = action['y']
            pyautogui.moveTo(x, y, duration=0.1 / acceleration_factor)
        elif action['type'] == 'click':
            css_selector = action.get('css_selector')
            frame_path = action.get('framepath', [])
            if css_selector:
                try:
                    switch_to_frame(frame_path)
                    # Attendre que l'élément soit cliquable
                    element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector)))
                    # Scroll l'élément dans la vue
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                    time.sleep(0.5)
                    element.click()
                    print(f"Clic effectué sur l'élément avec le sélecteur CSS '{css_selector}'.")
                except Exception as e:
                    print(f"Erreur lors du clic sur l'élément avec le sélecteur CSS '{css_selector}': {e}")
                    try:
                        # Utiliser JavaScript pour cliquer sur l'élément
                        driver.execute_script("arguments[0].click();", element)
                        print(f"Clic effectué via JavaScript sur l'élément avec le sélecteur CSS '{css_selector}'.")
                    except Exception as js_e:
                        print(f"Erreur lors du clic via JavaScript sur l'élément avec le sélecteur CSS '{css_selector}': {js_e}")
                        # Utiliser pyautogui en dernier recours
                        pyautogui.click()
                finally:
                    driver.switch_to.default_content()
            else:
                # Si aucun sélecteur CSS n'est fourni, effectuer un clic via pyautogui
                pyautogui.click()
        elif action['type'] == 'change':
            css_selector = action.get('css_selector')
            value = action.get('value')
            frame_path = action.get('framepath', [])
            if css_selector and value is not None:
                try:
                    switch_to_frame(frame_path)
                    # Attendre que l'élément soit présent
                    element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))
                    element.clear()
                    element.send_keys(value)
                    print(f"Valeur '{value}' saisie dans l'élément avec le sélecteur CSS '{css_selector}'.")
                except Exception as e:
                    print(f"Erreur lors de la modification de l'élément avec le sélecteur CSS '{css_selector}': {e}")
                    # Utiliser pyautogui en dernier recours
                    pyautogui.typewrite(value, interval=0.05 / acceleration_factor)
                finally:
                    driver.switch_to.default_content()
            else:
                print(f"Sélecteur CSS ou valeur manquant dans l'action : {action}")
        elif action['type'] == 'navigate':
            direction = action.get('direction')
            url = action.get('url')
            if direction == 'load' and url:
                try:
                    print(f"Navigation détectée vers l'URL : {url}")
                    # Attendre que l'URL change
                    wait.until(EC.url_to_be(url))
                    print(f"Nouvelle page chargée : {url}")
                    last_url = url
                except Exception as e:
                    print(f"Erreur lors de la navigation vers l'URL '{url}': {e}")
            elif direction == 'unload' and url:
                print(f"Déchargement de la page actuelle : {url}")
        else:
            print(f"Type d'action inconnu : {action['type']}")

        # Vérifier si une navigation a eu lieu en dehors des événements enregistrés
        current_url = driver.current_url
        if current_url != last_url:
            print(f"Changement d'URL détecté : {current_url}")
            last_url = current_url
            # Attendre que la nouvelle page soit chargée
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                print(f"Nouvelle page chargée : {current_url}")
            except Exception as e:
                print(f"Erreur lors de l'attente de la nouvelle page : {e}")

        # Petit délai après chaque action pour simuler le comportement humain
        time.sleep(0.1 / acceleration_factor)

    # Attendre que l'utilisateur appuie sur Entrée pour fermer le navigateur
    try:
        input("Appuyez sur Entrée pour arrêter la reproduction et fermer le navigateur...\n")
    except KeyboardInterrupt:
        pass

    # Fermer le navigateur
    driver.quit()
