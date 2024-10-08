import time
import threading
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
import pyautogui
from selenium.webdriver.common.by import By

# Délais d'attente
PAGE_LOAD_WAIT = 2  # Temps d'attente pour le chargement complet de la page
URL_CHANGE_WAIT = 1  # Temps d'attente après un changement d'URL
EVENT_RETRIEVE_INTERVAL = 0.005  # Intervalle de temps pour récupérer les événements
MOUSE_TRACK_INTERVAL = 0.05  # Intervalle de temps pour suivre la position de la souris

# Paramètre pour activer/désactiver le suivi de la position de la souris
log_positions = False  # Définir sur True pour activer le suivi

# Configuration du WebDriver
driver = webdriver.Firefox(service=FirefoxService())  # Assurez-vous que geckodriver est dans votre PATH

# Maximiser et positionner la fenêtre du navigateur pour des coordonnées cohérentes
driver.set_window_position(0, 0)
driver.maximize_window()

# Naviguer vers l'URL spécifiée
url = "https://www.bing.com/images/create"
driver.get(url)

# Attendre que la page se charge complètement
time.sleep(PAGE_LOAD_WAIT)

# Ouverture du fichier de log
log_file = open("actions_log.txt", "w", encoding="utf-8")

def log_action(action, event_time=None):
    if event_time is None:
        event_time = time.time()
    log_file.write(f"{event_time} - {action}\n")
    log_file.flush()

# Code JavaScript pour injecter les écouteurs d'événements
event_listener_script = """
(function() {
    function getClickableParent(element) {
        while (element && element !== document.body) {
            if (isClickable(element)) {
                return element;
            }
            element = element.parentElement;
        }
        return null;
    }

    function isClickable(element) {
        const tag = element.tagName.toLowerCase();
        const clickableTags = ['a', 'button', 'input', 'select', 'textarea', 'label', 'area'];
        if (clickableTags.includes(tag)) {
            return true;
        }
        if (element.hasAttribute('onclick')) {
            return true;
        }
        if (element.tabIndex >= 0) {
            return true;
        }
        if (getComputedStyle(element).cursor === 'pointer') {
            return true;
        }
        return false;
    }

    function getCssSelector(el) {
        if (!(el instanceof Element))
            return;
        var path = [];
        while (el.nodeType === Node.ELEMENT_NODE) {
            var selector = el.nodeName.toLowerCase();
            if (el.id) {
                selector += '#' + el.id;
                path.unshift(selector);
                break;
            } else {
                var sib = el, nth = 1;
                while ((sib = sib.previousElementSibling) != null) {
                    if (sib.nodeName.toLowerCase() == selector)
                        nth++;
                }
                if (nth != 1)
                    selector += ":nth-of-type(" + nth + ")";
            }
            path.unshift(selector);
            el = el.parentNode;
        }
        return path.join(" > ");
    }

    function addEventListeners(win) {
        // Éviter d'ajouter plusieurs fois les écouteurs d'événements
        if (win.hasEventListenersAdded) return;
        win.hasEventListenersAdded = true;

        if (!win.eventLog) win.eventLog = [];

        win.document.addEventListener('click', function(event) {
            var element = getClickableParent(event.target) || event.target;
            var info = {
                time: Date.now(),
                type: 'click',
                tag: element.tagName,
                id: element.id || 'N/A',
                class: element.className || 'N/A',
                cssSelector: getCssSelector(element),
                framePath: getFramePath(win)
            };
            win.eventLog.push(info);
            // Délai pour s'assurer que l'événement est loggé avant la navigation
            setTimeout(function() {}, 100);
        }, true);

        win.document.addEventListener('change', function(event) {
            var element = event.target;
            var info = {
                time: Date.now(),
                type: 'change',
                tag: element.tagName,
                id: element.id || 'N/A',
                class: element.className || 'N/A',
                cssSelector: getCssSelector(element),
                value: element.value || 'N/A',
                framePath: getFramePath(win)
            };
            win.eventLog.push(info);
        }, true);

        // Observer les mutations pour gérer le contenu dynamique
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            // Si le nœud ajouté est une frame, ajouter des écouteurs
                            if (node.tagName.toLowerCase() === 'iframe' || node.tagName.toLowerCase() === 'frame') {
                                try {
                                    addEventListeners(node.contentWindow);
                                } catch (e) {
                                    // Impossible d'accéder aux frames cross-origin
                                }
                            }
                        }
                    });
                }
            });
        });
        observer.observe(win.document.body, { childList: true, subtree: true });

        // Récursivement ajouter des écouteurs d'événements aux iframes de même origine
        for (var i = 0; i < win.frames.length; i++) {
            try {
                addEventListeners(win.frames[i]);
            } catch (e) {
                // Impossible d'accéder aux frames cross-origin
            }
        }
    }

    function getFramePath(win) {
        var path = [];
        var currentWin = win;
        while (currentWin !== window.top) {
            var idx = Array.prototype.indexOf.call(currentWin.parent.frames, currentWin);
            path.unshift(idx);
            currentWin = currentWin.parent;
        }
        return path;
    }

    addEventListeners(window);
})();
"""

def inject_event_listener_script():
    driver.execute_script(event_listener_script)
    # Injecter dans les iframes accessibles
    def inject_into_frames():
        frames = driver.find_elements(By.TAG_NAME, "iframe") + driver.find_elements(By.TAG_NAME, "frame")
        for frame in frames:
            try:
                driver.switch_to.frame(frame)
                driver.execute_script(event_listener_script)
                inject_into_frames()  # Injection récursive dans les frames imbriquées
                driver.switch_to.parent_frame()
            except Exception as e:
                print(f"Impossible d'injecter le script dans la frame : {e}")
                driver.switch_to.default_content()
    inject_into_frames()
    driver.switch_to.default_content()

# Injection initiale du script d'écoute des événements
inject_event_listener_script()

# Fonction pour récupérer et logger les événements
def retrieve_events():
    last_timestamp = 0
    last_url = driver.current_url
    while True:
        try:
            if not driver.service.is_connectable():
                break

            # Vérifier si l'URL a changé (navigation vers une nouvelle page)
            current_url = driver.current_url
            if current_url != last_url:
                print(f"Changement d'URL détecté : {last_url} -> {current_url}. Réinjection du script.")
                last_url = current_url
                time.sleep(URL_CHANGE_WAIT)
                # Réinjecter le script d'écoute des événements
                inject_event_listener_script()

            # Récupérer les événements de la fenêtre principale et des frames accessibles
            events = []
            def retrieve_events_from_window(win):
                win_events = win.execute_script("""
                    return window.eventLog || [];
                """)
                # Effacer les événements récupérés
                win.execute_script("window.eventLog = [];")
                events.extend(win_events)
                frames = win.find_elements(By.TAG_NAME, "iframe") + win.find_elements(By.TAG_NAME, "frame")
                for frame in frames:
                    try:
                        win.switch_to.frame(frame)
                        retrieve_events_from_window(win)
                        win.switch_to.parent_frame()
                    except:
                        win.switch_to.default_content()
            retrieve_events_from_window(driver)

            for event in events:
                # Convertir le timestamp en secondes
                event_time = event['time'] / 1000.0  # millisecondes en secondes
                if event_time > last_timestamp:
                    last_timestamp = event_time
                    action = f"Event Type: {event['type']} - Tag: {event['tag']} - ID: {event['id']} - Class: {event['class']} - CSS Selector: {event['cssSelector']} - Frame Path: {event.get('framePath', [])}"
                    if 'value' in event:
                        action += f" - Value: {event.get('value', '')}"
                    log_action(action, event_time)
            time.sleep(EVENT_RETRIEVE_INTERVAL)
        except Exception as e:
            print(f"Erreur dans retrieve_events: {e}")
            break

# Démarrer le thread de récupération des événements
event_thread = threading.Thread(target=retrieve_events)
event_thread.daemon = True
event_thread.start()

# Fonction facultative pour suivre la position de la souris
def track_mouse_position():
    while True:
        try:
            if not driver.service.is_connectable():
                break
            x, y = pyautogui.position()
            log_action(f"Mouse Position: x={x}, y={y}")
            time.sleep(MOUSE_TRACK_INTERVAL)
        except:
            break

# Démarrer le thread de suivi de la souris si log_positions est True
if log_positions:
    mouse_thread = threading.Thread(target=track_mouse_position)
    mouse_thread.daemon = True
    mouse_thread.start()

# Attendre que l'utilisateur appuie sur Entrée pour arrêter l'enregistrement
try:
    input("Appuyez sur Entrée pour arrêter l'enregistrement et fermer le navigateur...\n")
except KeyboardInterrupt:
    pass

# Nettoyage
log_file.close()
driver.quit()
