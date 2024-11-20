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

    os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2000"

    st.sidebar.image("assets/no_back_logo.png", output_format="PNG")
    #st.logo("assets/icon_ai.jpg",size='large')

    # load configuration
    load_config()

    # defining the allowed data sources in the variable options
    options = st.session_state["config"]["data_sources"]

    


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
        #"field_filter": field_filter,
        "deep_search": deep_search,
        "chat_history": str(
            "## Chat History: \n\n " + str(st.session_state["messages"])
        ),  # we keep track of the chat history,
        "use_history": False,  # this functionnality is in work in progress
        # "auto_job": auto_job,
        "return_chunks": True,
        "stream": True,
    }


    st.session_state["streamlit_config"] = streamlit_config
    
   
    if st.sidebar.toggle(
        "Enable audio recording 🎤",
        value=False,
        help="You can allow audio recording (will make a recording button appear) and then it will be immediately transcribed.",
    ):
        audio = st.audio_input(
            "Start recording"
        )
        
        print("AUDIO: ", audio)
        print("TYPE: ", type(audio))

        if audio:
            print("AUDIO: ", audio)
            print("AUDIO RECORDED !")
            # Save the recorded audio to a WAV file
            #if the temp directory doesn't exist, create it
            if not os.path.exists("temp"):
                os.makedirs("temp")
                
            with open("temp/recorded_audio.wav", "wb") as f:
                f.write(audio.getbuffer())
                
            print("Audio saved as recorded_audio.wav")
            
            from src.aux_utils.transcription_utils import YouTubeTranscriber
            
            with st.spinner("Transcribing audio...🎤"):
                yt=YouTubeTranscriber()
                transcription=yt.transcribe("temp/recorded_audio.wav",method="groq")
                
            print("TRANSCRIPTION: ", transcription)
            
            #write the transcription in the chat messages
            
            #st.session_state["messages"].append("🎤 Audio recorded and transcribed: \n\n"+transcription)
            st.chat_message("assistant").write("🎤 Audio recorded and transcribed: \n\n"+transcription)
            # save the transcription in a json file
            # pdf = create_transcription_pdf(transcription)
            # from src.streamlit_app_utils import create_transcription_txt
            # txt = create_transcription_txt(transcription)
            # print("TXT FILE CREATED !")
            # st.session_state["uploaded_file"] = True

            # with open(txt, "rb") as f:
            #     data = f.read()

            st.toast("Transcription successfull !", icon="🎉")

            # st.download_button(
            #     label="Download transcription as txt",
            #     data=txt,
            #     file_name="transcription.txt",
            #     mime="text/plain",
            # )

            # st.session_state["transcription"] = str(txt)

            # audio = []

            # from src.streamlit_app_utils import show_submission_form
            # show_submission_form()


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

    # Get the query from the user
    query = st.chat_input("Please enter your question")

    # Process the query if submitted
    if query:
        st.session_state["current_query"] = query
        process_query(query, streamlit_config,rag_agent=rag_agent)


if __name__ == "__main__":
    main()