import json
import os

import streamlit as st

# from audiorecorder import audiorecorder
# load environment variables
from dotenv import load_dotenv

# custom imports
from src.streamlit_tind_utils import (
    display_chat_history,
    initialize_session_state,
    load_config,
    process_query,
)


load_dotenv()
    
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"

load_config()

temperature=1

# a dropdown menu to select the origin of the generated women
origin=st.sidebar.selectbox("Select the origin of the generated women",["European","Asian","African","Indian","Middle Eastern","Latin American","Native American"])
# Load configuration from config.yaml file and initialize session state

initialize_session_state()

# Display the chat history
display_chat_history()

# Here are defined all usefull variables for the chatbot
streamlit_config = {
    "temperature": temperature,
    "chat_history": str(
        "## Chat History: \n\n " + str(st.session_state["messages"])
    ),  # we keep track of the chat history,
    "origin":origin
}


st.session_state["streamlit_config"] = streamlit_config

# Get the query from the user
query = st.chat_input("Please enter your question")


# Process the query if submitted
if query:
    st.session_state["current_query"] = query
    process_query(query, streamlit_config)


