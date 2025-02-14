import streamlit as st
from annotated_text import annotated_text

def create_annotations(text, chunks):
    """
    Crée une liste d'éléments pour annotated_text en utilisant les indices des chunks.
    
    Args:
        text (str): Le texte complet du document
        chunks (list): Liste des objets Document avec leurs métadonnées
    
    Returns:
        list: Liste d'éléments pour annotated_text
    """
    elements = []
    last_pos = 0
    
    # Collecte des positions depuis les métadonnées
    positions = []
    for chunk in chunks:
        start_index = chunk.metadata.get('start_index', -1)
        end_index = chunk.metadata.get('end_index', -1)
        if start_index != -1 and end_index != -1:
            positions.append((start_index, end_index, chunk))
    
    # Tri des positions par index de début
    positions.sort(key=lambda x: x[0])
    
    # Construction des annotations
    for start, end, chunk in positions:
        # Texte avant le chunk
        if start > last_pos:
            elements.append(text[last_pos:start])
            
        # Le chunk lui-même (texte surligné)
        chunk_text = text[start:end]
        chunk_id = chunk.metadata.get('_id', 'no-id')  # Identifiant unique du chunk
        elements.append((chunk_text, chunk_id, "#282c34"))  # Fond gris élégant et moderne
        
        last_pos = end
    
    # Ajout du texte restant après le dernier chunk
    if last_pos < len(text):
        elements.append(text[last_pos:])
    
    return elements

def show_full_documents():
    st.set_page_config(page_title="Sources complètes", layout="wide")
    st.title("📜 Documents Sources Complets")

    if 'sources_dict' not in st.session_state:
        st.warning("Aucun document disponible")
        return

    for path, meta in st.session_state.sources_dict.items():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            chunks = meta.get('Chunks', [])
            elements = create_annotations(full_text, chunks)

            with st.expander(f"{meta['Emoji']} {meta['Filename']}"):
                annotated_text(*elements)
                st.caption(f"Chemin: {path} | Chunks trouvés: {len(chunks)}")

        except Exception as e:
            st.toast(f"Erreur avec {path}: {str(e)}", icon="❌")
            st.error(f"Erreur technique: {str(e)}")

if __name__ == "__main__":
    show_full_documents()