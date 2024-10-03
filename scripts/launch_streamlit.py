import os
import sys

if __name__ == "__main__":
    # Obtenez le chemin temporaire où les fichiers sont extraits
    if getattr(sys, 'frozen', False):
        # L'exécutable est en cours d'exécution
        base_path = sys._MEIPASS
    else:
        # Le script est en cours d'exécution
        base_path = os.path.abspath(".")

    # Chemin vers le script Streamlit
    streamlit_script = os.path.join(base_path, 'scripts', 'streamlit_app.py')
    
    # Lancez Streamlit
    os.system(f"streamlit run {streamlit_script}")