import streamlit as st
import os

# Configuration
st.set_page_config(page_title="Visualiseur de Fichier Local", page_icon="📄")

# Chemin du fichier
file_path = r"C:\Users\Yanis\Documents\RAG\data\politique\_MZ5mN_HMXg.txt"
filename = os.path.basename(file_path)

def get_file_content():
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Erreur lors de la lecture du fichier: {str(e)}"

# URL fixe avec port 8501 (port par défaut de Streamlit)
local_url = "http://localhost:8501/?view_file=true"

st.title("Visualiseur de Fichier Local")

# Récupérer les paramètres de requête
query_params = st.experimental_get_query_params()

if "view_file" in query_params:
    st.subheader(f"Contenu du fichier : {filename}")
    content = get_file_content()
    st.text_area("", content, height=400)
else:
    # Créer un lien HTML personnalisé
    link_html = f'''
    <a href="{local_url}" target="_blank" style="text-decoration: none;">
        <div style="
            display: inline-flex;
            align-items: center;
            background-color: #f0f2f6;
            padding: 8px 12px;
            border-radius: 5px;
            border: 1px solid rgba(49, 51, 63, 0.2);
            ">
            <span style="font-size: 18px; margin-right: 8px;">📄</span>
            <span style="font-size: 16px; color: #262730;">Voir {filename}</span>
        </div>
    </a>
    '''
    st.markdown(link_html, unsafe_allow_html=True)
    
    # Aperçu du contenu
    content = get_file_content()
    st.text_area("Aperçu du contenu du fichier", content, height=200)