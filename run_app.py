# run_app.py
import os
from streamlit.web import cli as stcli
import streamlit
import sys

if __name__ == "__main__":
    # Obtenir le chemin absolu vers streamlit_app.py
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "streamlit_app.py")
    script_path = os.path.abspath(script_path)
    
    # Vérifier que le fichier existe
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Cannot find {script_path}")
        
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())