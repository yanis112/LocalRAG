import os
import streamlit as st
from dotenv import load_dotenv
from src.main_utils.streamlit_app_utils import (
    display_chat_history,
    initialize_session_state,
    load_config,
    process_query,
)



def main():
    load_dotenv()

    os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "5000"

    st.sidebar.image("assets/no_back_logo.png", output_format="PNG")
    

    # load configuration
    load_config()

    # defining the allowed data sources in the variable options
    # options = st.session_state["config"]["data_sources"]


    # Define sidebar parameters
    st.sidebar.header("Search Settings")
    
    # Define sidebar parameters
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
    
    #selectbox for LLM
    formatted_options = [f"{model} ({provider})" for model, provider in st.session_state["config"]["models_dict"].items()]
    llm_choice=st.sidebar.selectbox(
        "Select the LLM model",
        options=formatted_options,
        index=0,
        help="Select the LLM model you want to use for the answer generation."
    )
    
    if llm_choice:
        selected_model = llm_choice.split(" (")[0]
        print("Selected model: ", selected_model)
        selected_provider = llm_choice.split(" (")[1].rstrip(")")
        print("Selected provider: ", selected_provider)

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
    #st.sidebar.header("LLM Settings")
   
    # Load configuration from config.yaml file and initialize session state
    initialize_session_state()
    
    st.sidebar.header("Audio Recording/Uploading")

    # Display the chat history
    display_chat_history()
    
    #load the RAG agent only once (not at each query)
    @st.cache_resource
    def load_rag_agent():
        print("RAG AGent not in session state, loading it...")
        from src.main_utils.generation_utils_v2 import RAGAgent
        agent = RAGAgent(default_config=st.session_state["config"],config=st.session_state["streamlit_config"])
        return agent
    
    # Here are defined all usefull variables for the chatbot
    streamlit_config = {
        "cot_enabled": use_cot,
        "deep_search": deep_search,
        "chat_history": str(
            "## Chat History: \n\n " + str(st.session_state["messages"])
        ),  # we keep track of the chat history,
        "model_name": selected_model,
        "llm_provider": selected_provider,
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
        
        audio = st.audio_input("Start recording")
        
        if audio and "audio_transcription" not in st.session_state:
            from src.main_utils.streamlit_app_utils import process_audio_recording
            transcription = process_audio_recording(audio)
            st.session_state["audio_transcription"] = transcription
        
        
         

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
        from src.main_utils.streamlit_app_utils import handle_uploaded_file
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
            from src.main_utils.streamlit_app_utils import show_submission_form
            show_submission_form()
            
    # Add a button to clear the chat
    if st.sidebar.button(
        label="Clear Chat 🧹",
        help="Clear the chat history on the visual interface 🧹",
    ):
        from src.main_utils.streamlit_app_utils import clear_chat_history
        clear_chat_history()
        
    #we define the rag agent 
    rag_agent=load_rag_agent()
    
    #modify the rag agent config attribute to add the streamlit config
    rag_agent.config=st.session_state["streamlit_config"]

    # Get the query from the user
    query = st.chat_input("Please enter your question")

    # Process the query if submitted
    if query:
        st.session_state["current_query"] = query
        process_query(query, streamlit_config,rag_agent=rag_agent)


if __name__ == "__main__":
    main()