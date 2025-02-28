from jobspy import scrape_jobs
import os
import torch
import streamlit as st
from dotenv import load_dotenv
from src.main_utils.streamlit_app_utils import (
    display_chat_history,
    initialize_session_state,
    load_config,
    process_query,
)



# Désactive l’itération sur l'attribut __path__ des classes torch
torch.classes.__path__ = []

def main():
    load_dotenv()

    st.set_page_config(
    page_title="Main Page",
    page_icon="👋",
)
    
    def load_css(file_path: str):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    css_path = "custom_css/stunning_button.css"
    load_css(css_path)  # Add this line to load the CSS

    os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "5000"

    st.sidebar.image("assets/logo_v5.png", output_format="PNG")

    # load configuration
    load_config()
    
    with st.container():
        display_chat_history()

    # defining the allowed data sources in the variable options
    # options = st.session_state["config"]["data_sources"]
    
    st.sidebar.page_link(page="pages/sources_page.py", label="Data Sources",icon="📜",help="Acces to the sources used by the AI to answer")

    # Define sidebar parameters
    st.sidebar.header("Search Settings")
    
    # pg = st.navigation([st.Page("streamlit_app.py"),st.Page("sources_page.py")])
    # pg.run()

    # Define sidebar parameters
    # use_cot = st.sidebar.toggle(
    #     "Enable Justification 📝",
    #     value=False,
    #     key="justification_toggle",
    #     help="Enable or disable the use of Chain-of-Thought prompting. This will affect the LLM answer: longer, interpretable, less prown to hallucination, \
    #         step-by-step answer with document citation, but longer.... If disabled, the answer will be direct and synthetic.",
    # )

    deep_search = st.sidebar.toggle(
        "Enable Deep Search 🔥",
        value=False,
        key="deep_search_toggle",
        help="Enable or disable the deep search. This will make a search divided in several steps, fit to solve complex queries, more powerfull but longer...",
    )

    # selectbox for LLM
    formatted_options = [
        f"{model} ({provider})"
        for model, provider in st.session_state["config"]["models_dict"].items()
    ]
    llm_choice = st.sidebar.selectbox(
        "Select the LLM model",
        options=formatted_options,
        index=0,
        help="Select the LLM model you want to use for the answer generation.",
        key="llm_selectbox",
    )
    
    #slider for temperature
    temperature=st.sidebar.slider("Temperature 🌡️",min_value=0.0,max_value=1.0,step=0.1,value=0.8,help="The temperature is a hyperparameter that controls the randomness of the predictions that the model makes. Lower temperatures make the model more confident, but also more conservative. Higher temperatures make the model more creative, but also more risky.")
 

    selected_model = llm_choice.split(" (")[0]
    selected_provider = llm_choice.split(" (")[1].rstrip(")")

    # Define sidebar parameters
    # st.sidebar.header("Data Sources")
    # field_filter = st.sidebar.multiselect(
    #     "Select the data fields you want to search in",
    #     options=options.keys(),
    #     # by default all except chatbot_history
    #     default=list(options.keys()),
    #     format_func=lambda x: options[x],
    # )

    # Define sidebar parameters
    # st.sidebar.header("LLM Settings")

    # Load configuration from config.yaml file and initialize session state
    initialize_session_state()

    if "external_resources_list" not in st.session_state:
        st.session_state["external_resources_list"] = []

    st.sidebar.header("Audio Recording/Uploading")
    
    #create a container that will be used to display the chat messages later
    
    # chat_container=st.container()
    # st.session_state["chat_container"]=chat_container

    # Display the chat history
    #display_chat_history()

    # load the RAG agent only once (not at each query)
    @st.cache_resource
    def load_rag_agent():
        print("RAG AGent not in session state, loading it...")
        from src.main_utils.generation_utils_v2 import RAGAgent

        agent = RAGAgent(
            default_config=st.session_state["config"],
            config=st.session_state["streamlit_config"],
        )
        return agent

    # Here are defined all usefull variables for the chatbot
    streamlit_config = {
        "cot_enabled": False,
        "deep_search": deep_search,
        "chat_history": str(
            "## Chat History: \n\n " + str(st.session_state["messages"])
        ),  # we keep track of the chat history,
        "model_name": selected_model,
        "llm_provider": selected_provider,
        "temperature": temperature,
        "use_history": False,  # this functionnality is in work in progress
        "return_chunks": True,
        "stream": True,
    }

    st.session_state["streamlit_config"] = streamlit_config

    if st.sidebar.toggle(
        "Enable audio recording 🎤",
        value=False,
        help="You can allow audio recording (will make a recording button appear) and then it will be immediately transcribed.",
    ):
        audio_bytes = st.audio_input(
            "Start recording"
        )  # Renamed 'audio' to 'audio_bytes' for clarity

        # Initialize session state for processed audio data if it doesn't exist
        if "processed_audio_data" not in st.session_state:
            st.session_state.processed_audio_data = None

        # Check if the audio data has changed
        if audio_bytes != st.session_state.processed_audio_data:
            # Audio data has changed (or it's the first recording)
            st.session_state.processed_audio_data = (
                audio_bytes  # Update processed audio data
            )

            if audio_bytes:
                print("NEW AUDIO RECORDING DETECTED!")  # Added print for debugging
                from src.main_utils.streamlit_app_utils import process_audio_recording

                transcription = process_audio_recording(
                    audio_bytes
                )  # Use audio_bytes here
                st.session_state["audio_transcription"] = transcription
                st.success("Audio transcribed!")  # Added a success message
            # else:
            #     # Audio recording was potentially cleared (though st.audio_input doesn't have a clear button directly)
            #     st.session_state.pop("audio_transcription", None) # Clear previous transcription if audio is removed
            #    # st.write("No audio recording present.") # Inform user if audio is removed

        # else:
        #     # Audio data is the same as last processed, no need to re-transcribe
        #     if audio_bytes:
        #         st.write("Audio already transcribed (no changes detected).")
        #     else:
        #         st.write("Start recording audio to transcribe.") # Initial message when no audio recorded yet

    else:  # If audio recording is toggled off
        # st.session_state.pop("audio_transcription", None) # Optionally clear transcription when recording is disabled
        st.session_state.pop(
            "processed_audio_data", None
        )  # Clear processed audio data when recording is disabled
        # st.write("Audio recording disabled. Toggle to enable.") # Inform user recording is disabled

    uploaded_files = st.sidebar.file_uploader(
        label="Choose a file (audio or image)",
        type=["mp3", "jpg", "png", "wav", "jpeg", "m4a", "mp4", "pdf"],
        key="file_uploader",
        accept_multiple_files=True,
        help="You can upload any audio file and it will be immediately transcribed (with all speakers identification 🔥). You can also upload an image file and it will be OCRized (image-to-text). Delete previous audio or image manually before uploading a new one ! WARNING: transcriptions or ocerized text isn't used by the chatbot to provide answers. ",
    )

    # Get the filenames of the currently uploaded files
    current_filenames = [file.name for file in uploaded_files] if uploaded_files else []

    # Initialize session state for processed filenames if it doesn't exist
    if "processed_filenames" not in st.session_state:
        st.session_state["processed_filenames"] = []

    # Check if the filenames of the uploaded files have changed
    if current_filenames != st.session_state["processed_filenames"]:
        # Filenames have changed (or it's the first upload)
        st.session_state["processed_filenames"] = (
            current_filenames  # Update processed filenames
        )

        if (
            uploaded_files
        ):  # This condition is still useful to check if there are files to process
            st.toast(f"{len(uploaded_files)} file(s) uploaded successfully!", icon="✅")
            print("FILES UPLOADED!")
            # st.session_state["uploaded_files_processed"] = True # Removed this line
            from src.main_utils.streamlit_app_utils import (
                handle_multiple_uploaded_files,
            )

            handle_multiple_uploaded_files(uploaded_files, parallel=False)
        elif (
            "uploaded_files" in st.session_state
        ):  # Handle the case where files were previously uploaded but now are removed.
            st.session_state.pop(
                "uploaded_files"
            )  # Optionally remove uploaded_files from session state if needed.
            st.write(
                "No files currently uploaded."
            )  # Inform the user that files were removed.

    # we define the rag agent
    rag_agent = load_rag_agent()

    diarization_enabled = st.sidebar.toggle(
        "Enable diarization 🗣️",
        value=False,
        key="diarization_toggle",
        help="Enable or disable the speaker diarization. This will allow to identify the speakers in the audio recording while transcribing it.",
    )

    st.session_state["diarization_enabled"] = diarization_enabled

    if st.sidebar.button(
        label="Index 🗃️",
        help="Index all resources that have been upploaded.",
        key="button1",
    ):
        from src.main_utils.link_gestion import ExternalKnowledgeManager
        from stqdm import stqdm

        resource_manager = ExternalKnowledgeManager(
            config=rag_agent.merged_config, client=rag_agent.client
        )
        # the texts/resources are contained in a list in the session state "external_resources_list"
        for text in stqdm(st.session_state["external_resources_list"]):
            resource_manager.extract_rescource(text=text)
            resource_manager.index_rescource()
            
        #we empty the list of external resources list
        st.session_state["external_resources_list"] = []

        # we stop the process here
        st.toast("New rescource indexed !", icon="🎉")

    # Add a button to clear the chat
    if st.sidebar.button(
        label="Clear Chat 🧹",
        help="Clear the chat history on the visual interface 🧹",
        key="button2",
    ):
        from src.main_utils.streamlit_app_utils import clear_chat_history
        clear_chat_history()
        
    if st.sidebar.button(
        label="Export to Notion 📝",
        help="Export last chat messages to a Notion page 📝",
        key="button3",
    ):
        from src.main_utils.streamlit_app_utils import export_to_notion
        #on delete toute ce qu'il y a comme texte entre les balises <think> et </think> dans le message si elles existent
        raw_answer=st.session_state["messages"][-1]["content"]
        #using regex to delete the text between the tags <think> and </think>
        import re
        raw_answer=re.sub(r'<think>.*</think>', '', raw_answer)
        export_to_notion(text=raw_answer)
        
    

    if st.sidebar.toggle("Qrcode", value=False, key="qrcode_toggle"):
        from src.main_utils.streamlit_app_utils import create_qrcode

        create_qrcode()
        st.sidebar.image("assets/qr_code_2.png", output_format="PNG")

    # modify the rag agent config attribute to add the streamlit config
    rag_agent.config = st.session_state["streamlit_config"]

    # Get the query from the user
    
    
    # choice = st.pills(
    #     label="test",
    #     default="Search",
    #     options=[
    #         "🔍",
    #         "Search",
    #         "Search for the answer to your question",
    #         "primary",
    #     ],
    #     key="fixed_pills",
    #     label_visibility="collapsed"
    # )
    query = st.chat_input(placeholder="Type your question here...")

    # Process the query if submitted
    if query:
        st.session_state["current_query"] = query
        process_query(query, streamlit_config, rag_agent=rag_agent)
    
   


if __name__ == "__main__":
    main()
