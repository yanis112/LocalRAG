import streamlit as st
import pathlib
from glob import glob

def load_css_files():
    css_files = glob("stunning_button/*.css")
    
    
    for css_file in css_files:
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css_files()

st.sidebar.button("Click me", key="button1")
st.sidebar.toggle("Click me", key="toggle1")