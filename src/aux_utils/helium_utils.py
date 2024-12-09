from helium import *
import time
start_chrome(url="https://auth.centrale-marseille.fr/cas/login?service=https%3A%2F%2Fwmail.centrale-marseille.fr%2F")
write('ylabeyrie', into='Identifiant:')
write('Yanis.Labeyrie@06', into='Mot de passe:')
#cliquer sur le bouton pour se connecter
click('Se connecter')
#attendre 5 secondes
time.sleep(5)
#clicker sur le bouton "Rafraîchir"
click("Rafraîchir")