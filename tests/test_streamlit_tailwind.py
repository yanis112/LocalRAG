import streamlit as st
import st_tailwind as tw

st.set_page_config("Streamlit Tailwind Examples")

def main():
    tw.initialize_tailwind()

    tw.write("Exemples de boutons modernes avec Daisy UI", classes="text-3xl font-bold mb-6")
    st.button("Normal Button")  
    # Exemple 1: Boutons avec dégradé et effet hover

    tw.button("Glass Buton", classes="btn glass")

  
if __name__ == "__main__":
    main()