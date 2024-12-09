import socket
import streamlit as st

def get_local_ip():
    try:
        # Crée une connexion socket temporaire pour obtenir l'IP locale
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

# Obtenir l'IP et construire l'URL
ip_address = get_local_ip()
port = 8501  # Port par défaut de Streamlit
network_url = f"http://{ip_address}:{port}"

# Afficher l'URL
st.write("Network URL:", network_url)