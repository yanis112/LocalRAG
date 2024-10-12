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
    process_query_v2,
)

load_dotenv()

os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"

# load configuration
load_config()




# a slider to set the temperature of the LLM
temperature = st.sidebar.slider(
    "Set the temperature of the LLM 🔥",
    0.0,
    3.0,
    1.0,
    step=0.1,
    help="Set the temperature of the LLM. A higher temperature will make the LLM more creative and less deterministic and factual, but also more prone to hallucination. A lower temperature will make the LLM more deterministic and less creative.",
)

# a dropdown menu to select the origin of the generated women
origin=st.sidebar.selectbox("Select the origin of the generated women",["European","Asian","African","Indian","Middle Eastern","Latin American","Native American"])
# Load configuration from config.yaml file and initialize session state
load_config()
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
    process_query_v2(query, streamlit_config)


