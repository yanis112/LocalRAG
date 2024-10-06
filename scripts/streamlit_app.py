import json
import os

import streamlit as st
#from audiorecorder import audiorecorder

# load environment variables
from dotenv import load_dotenv

# custom imports
from src.streamlit_app_utils import (
    clear_chat_history,
    create_transcription_txt,
    display_chat_history,
    feedback_dialog,
    handle_uploaded_file,
    initialize_session_state,
    load_config,
    process_query_v2,
    query_suggestion,
    show_submission_form,
    transcribe_audio,
)
from streamlit_lottie import st_lottie
from st_copy_to_clipboard import st_copy_to_clipboard
from src.utils import save_question_answer_pairs

load_dotenv()

os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"

# Initialize the chat interface
st.title(
    "LlamaNova 🦙",
    help="How to use Euranova's chatbot ? Just ask a query correctly formatted (ponctuation, capital letter for proper nouns,exact word spelling, ect..) \
    and the chatbot will find the most relevant documents to answer your question (5 documents) and cite the sources used after the answer. What you cannot expect from the chatbot ? This chatbot is not conversational \
    and does not have a memory of the previous interactions or messages, each query is independently processed from the previous ones. You can't ask anything apart from \
    a query/question, no greetings, no small talk, its a waste of time and resources !",
)


#st.sidebar.image("assets/logo_v5.png", output_format="PNG")


# load configuration
load_config()

# defining the allowed data sources in the variable options
options = st.session_state["config"]["data_sources"]

# Define the persist directory (where the embeddings are stored)
embedding_database = st.session_state["config"]["persist_directory"]

# Add a button to clear the chat
if st.sidebar.button(
    label="Clear Chat 🧹", help="Clear the chat history on the visual interface 🧹"
):
    clear_chat_history()
    
#add a copy to clipboard button to cpoy last chabot answer
if "messages" in st.session_state and len(st.session_state['messages']) > 0:
    print("MESSAGES: ", st.session_state['messages'])
    st_copy_to_clipboard(str(st.session_state['messages'][-1]['content']))

# Initialize theme state if it doesn't exist
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'
if 'theme_changed' not in st.session_state:
    st.session_state['theme_changed'] = False

# Function to toggle the theme
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.session_state.theme_changed = True

# Toggle button to switch themes
if st.sidebar.toggle("Dark Mode 🌞/🌜", value=False if st.session_state['theme'] == 'light' else True):
    if st.session_state.theme != 'dark':
        toggle_theme()
else:
    if st.session_state.theme != 'light':
        toggle_theme()

# Apply the theme
if st.session_state.theme == 'light':
    st._config.set_option('theme.base', 'light')
else:
    st._config.set_option('theme.base', 'dark')

# Check if the theme has been changed and reload the page if necessary
if st.session_state.theme_changed:
    st.session_state.theme_changed = False
    st.rerun()

# Define sidebar parameters
st.sidebar.header("Search Settings")
use_cot = st.sidebar.toggle(
    "Enable Justification 📝",
    value=False,
    help="Enable or disable the use of Chain-of-Thought prompting. This will affect the LLM answer: longer, interpretable, less prown to hallucination, \
        step-by-step answer with document citation, but longer.... If disabled, the answer will be direct and synthetic.",
)

deep_search = st.sidebar.toggle(
    "Enable Deep Search 🔥",
    value=False,
    help="Enable or disable the deep search. This will make a search divided in several steps, fit to solve complex queries, more powerfull but longer...",
)

# use_history = st.sidebar.toggle(
#     "Enable Chat History Use 📜",
#     value=False,
#     help="Enable or disable the chat history use by the chatbot. If enabled, the chatbot will use the chat history to provide more accurate answers. Else, the chatbot will ignore the chat history. Pay attention, if the chat history is too long, it can saturate the chatbot... dont forget to clean the chat history ! The history is only used if advanced search is enabled !",
# )

# Define sidebar parameters
st.sidebar.header("Data Sources")
field_filter = st.sidebar.multiselect(
    "Select the data sources you want to search in (default: all sources except chatbot history)",
    options=options.keys(),
    # by default all except chatbot_history
    default=list(options.keys()),
    format_func=lambda x: options[x],
)

# Define sidebar parameters
st.sidebar.header("LLM Settings")
# a slider to set the temperature of the LLM
temperature = st.sidebar.slider(
    "Set the temperature of the LLM 🔥",
    0.0,
    3.0,
    1.0,
    step=0.1,
    help="Set the temperature of the LLM. A higher temperature will make the LLM more creative and less deterministic and factual, but also more prone to hallucination. A lower temperature will make the LLM more deterministic and less creative.",
)


# Load configuration from config.yaml file and initialize session state
load_config()
initialize_session_state()

st.sidebar.header("Audio Recording/Uploading")

if st.sidebar.toggle(
    "Enable audio recording 🎤",
    value=False,
    help="You can allow audio recording (will make a recording button appear) and then it will be immediately transcribed.",
):
    audio = audiorecorder(
        start_prompt="Start recording", stop_prompt="Stop recording"
    )

    if len(audio) > 0:
        print("AUDIO: ", audio)
        print("AUDIO RECORDED !")
        transcribe_audio(audio)
        # get the transcrition from the session state
        transcription = st.session_state.messages[-1]["content"]
        # save the transcription in a json file
        # pdf = create_transcription_pdf(transcription)
        txt = create_transcription_txt(transcription)
        print("TXT FILE CREATED !")
        st.session_state["uploaded_file"] = True

        with open(txt, "rb") as f:
            data = f.read()

        st.toast("Transcription successfull !", icon="🎉")

        st.download_button(
            label="Download transcription as txt",
            data=txt,
            file_name="transcription.txt",
            mime="text/plain",
        )

        st.session_state["transcription"] = str(txt)

        audio = []

        show_submission_form()


uploaded_file = st.sidebar.file_uploader(
    label="Choose a file (audio or image)",
    type=["mp3", "jpg", "png", "wav", "jpeg", "m4a", "mp4", "pdf"],
    help="You can upload any audio file and it will be immediately transcribed (with all speakers identification 🔥). You can also upload an image file and it will be OCRized (image-to-text). Delete previous audio or image manually before uploading a new one ! WARNING: transcriptions or ocerized text isn't used by the chatbot to provide answers. ",
)

# if no file uploaded (or an uploaded file is removed), remove the uploaded_file session state
if uploaded_file is None:
    if "uploaded_file" in st.session_state:
        st.session_state.pop("uploaded_file")

if uploaded_file and "uploaded_file" not in st.session_state:
    st.toast("File uploaded successfully!", icon="✅")
    print("FILE UPLOADED !")
    st.session_state["uploaded_file"] = True
    output_file = handle_uploaded_file(uploaded_file)

    # si le type de output file c'est des bytes on met le download button
    if isinstance(output_file, bytes):
        st.download_button(
            label="Download transcription as txt",
            data=output_file,
            file_name="transcription.txt",
            mime="text/plain",
        )

        st.session_state["transcription"] = str(output_file.decode("utf-8"))

        show_submission_form()

#add a toogle for auto_job tool use
auto_job=st.sidebar.toggle(
    "Enable Auto Job Tool 📝",
    value=False,
    help="Enable or disable the use of the Auto Job Tool. This tool is used to help you to write a job application letter. You will be asked to provide some informations and the tool will generate a prompt for you to write the letter.",
)

# Display the chat history
display_chat_history()

streamlit_config = {
    "cot_enabled": use_cot,
    "field_filter": field_filter,
    "temperature": temperature,
    "deep_search": deep_search,
    "chat_history": str("## Chat History: \n\n "+str(st.session_state['messages'])), #we keep track of the chat history,
    "use_history": False, #this functionnality is in work in progress
    "auto_job": auto_job,
}

#print the chat history
#print("## Chat History: \n\n", st.session_state['messages'])

st.session_state["streamlit_config"] = streamlit_config

# Get the query from the user
query = st.chat_input("Please enter your question")


# Process the query if submitted
if query:
    st.session_state["current_query"] = query
    process_query_v2(query, streamlit_config)

if "feedback" not in st.session_state:
    if st.sidebar.button("Send Feedback"):
        feedback_dialog()
else:
    st.toast("Feedback sent successfully!", icon="✅")
    try:
        with open("feedback.json", "r") as f:
            try:
                data = json.load(f)
            except:
                data = []
    except FileNotFoundError:
        data = []

    data.append(
        {
            "question": st.session_state["current_query"],
            "answer": st.session_state.messages[-1]["content"],
            "feedback": st.session_state.feedback["feedback"],
            "comment": st.session_state.feedback["comment"],
        }
    )
    with open("feedback.json", "w") as f:
        json.dump(data, f)

    # remove feedback from session state
    st.session_state.pop("feedback")

