import io
import json
import os
import textwrap
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit_pills as stp
import yaml

# load environment variables
from dotenv import load_dotenv
from fpdf import FPDF
from pydub import AudioSegment

# custom imports
from src.agentic_rag_utils import (
    QueryBreaker,
)

from src.generation_utils import RAG_answer,advanced_RAG_answer
from src.knowledge_graph import KnowledgeGraph
from src.utils import (
    StructuredAudioLoaderV2,
    StructuredPDFOcerizer,
    token_calculation_prompt,
)

# Fonction pour sauvegarder le fichier audio en .wav
def save_audio_as_wav(uploaded_file, output_dir):
    """
    Saves the uploaded audio file as a WAV file in the specified output directory.

    Args:
        uploaded_file (str): The path to the uploaded audio file.
        output_dir (str): The directory where the WAV file will be saved.

    Returns:
        str: The path to the saved WAV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "input_file.wav")
    audio = AudioSegment.from_file(uploaded_file)
    audio.export(file_path, format="wav")
    return file_path


# def generate_qr_code(url):
#     """
#     Generates a QR code image for the given URL.

#     Parameters:
#     url (str): The URL to encode into the QR code.

#     Returns:
#     PIL.Image.Image: The generated QR code image.
#     """
#     qr = qrcode.QRCode(
#         version=1,
#         error_correction=qrcode.constants.ERROR_CORRECT_H,
#         box_size=10,
#         border=4,
#     )
#     qr.add_data(url)
#     qr.make(fit=True)
#     img = qr.make_image(fill="black", back_color="white")
#     return img


def load_config():
    """
    Loads the configuration from the config.yaml file and stores it in the session state.

    If the configuration is already loaded in the session state, it will not be loaded again.

    Returns:
        dict: The loaded configuration.
    """
    try:
        # safe load the config file
        with open("config/config.yaml") as file:
            config = yaml.safe_load(file)
        st.session_state["config"] = config

    except Exception as e:
        print("EXCEPTION IN LOAD CONFIG:", e)
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
            st.session_state["config"] = config
    return st.session_state["config"]


@st.fragment
def show_submission_form():
    with st.expander("Save in vectorstore options", expanded=True):
        # Champs à compléter
        with st.form(key="Save document in vectorstore"):
            st.write(
                "Dont press enter key when entering inputs, it will submit the form !"
            )
            st.session_state["document_title"] = st.text_input(
                "Name to give to the document in vectorstore"
            )
            st.session_state["small_description"] = st.text_input(
                "Explain in a few words what the document is about, date, subject, etc..."
            )
            submit = st.form_submit_button("Submit")
            if submit:
                print("Document title:", st.session_state["document_title"])
                # save the file in data/meeting_transcriptions with the given name
                with open(
                    "data/meeting_transcriptions/"
                    + st.session_state["document_title"]
                    + ".txt",
                    "w",
                ) as f:
                    f.write(
                        "DESCRIPTION: "
                        + st.session_state["small_description"]
                        + "\n\n"
                        + st.session_state["transcription"]
                    )
                    st.toast("Document saved in the database !", icon="🎉")


def initialize_session_state():
    """
    Initializes the session state by generating a QR code, setting up an empty list for messages,
    and creating a KnowledgeGraph object if it doesn't exist in the session state.

    Parameters:
        None

    Returns:
        None
    """
    # if "qr_code" not in st.session_state:
    #     STREAMLIT_URL = os.getenv("STREAMLIT_URL")
    #     qr_code = generate_qr_code(STREAMLIT_URL)
    #     byte_arr = io.BytesIO()
    #     qr_code.save(byte_arr, format="PNG")
    #     byte_arr = byte_arr.getvalue()
    #     st.session_state["qr_code"] = byte_arr

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_graph" not in st.session_state:
        st.session_state["knowledge_graph"] = KnowledgeGraph()


def transcribe_audio(audio):
    """
    Transcribes the given audio file.

    Args:
        audio: An audio file object.

    Returns:
        None
    """
    audio.export("temp/input_audio.wav", format="wav")
    st.write(
        f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds"
    )
    with st.spinner("Transcribing audio..."):
        transcriptor = StructuredAudioLoaderV2(
            file_path="temp/input_audio.wav", diarization=True, batch_size=4
        )
        transcription = transcriptor.transcribe_audio()
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Here is your transcribed audio 🔉📜 ! \n\n "
            + str(transcription),
        }
    )


def save_uploaded_pdf(uploaded_file, temp_dir="temp"):
    # Vérifier si le dossier temporaire existe, sinon le créer
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Créer un chemin de fichier dans le dossier temporaire
    file_path = os.path.join(temp_dir, uploaded_file.name)

    # Ouvrir un nouveau fichier en mode écriture binaire
    with open(file_path, "wb") as f:
        # Écrire les bytes du fichier PDF dans le nouveau fichier
        f.write(uploaded_file.getvalue())

    # Retourner le chemin du fichier sauvegardé
    return file_path


@st.fragment
def handle_uploaded_file(uploaded_file):
    temp_dir = "temp"
    extension = str(Path(uploaded_file.name).suffix)
    if extension in [".mp3", ".wav", ".m4a", ".mp4"]:
        file_path = save_audio_as_wav(uploaded_file, temp_dir)
        st.success(f"File saved as {file_path}")
        with st.spinner(
            "Transcribing the audio file... it should take less than 2 minutes 😜"
        ):
            transcriptor = StructuredAudioLoaderV2(
                file_path=file_path, batch_size=4, diarization=True
            )
            transcription = transcriptor.transcribe_audio()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Here is your transcribed audio 🔉📜 ! \n\n "
                + str(transcription),
            }
        )
        # pdf = create_transcription_pdf(transcription)
        txt = create_transcription_txt(transcription)
        print("TXT FILE CREATED !")
        st.session_state["uploaded_file"] = True

        with open(txt, "rb") as f:
            data = f.read()

        st.toast("Transcription successfull !", icon="🎉")

        print("DOWNLOAD BUTTON CREATED !")
        return data

    elif extension in [".png", ".jpg", ".jpeg"]:
        st.image(uploaded_file)
        #save the image in the temp folder
        uploaded_file.seek(0)
        with open("temp/" + uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        
        from src.unstructured_utils import UniversalImageLoader
        with st.spinner(
            "Analyzing the image... it should take less than 2 minutes 😜"
        ):
            #load the universal image loader
            structured_output=UniversalImageLoader().universal_extract(image_path="temp/" + uploaded_file.name)
            print("TYPE OF STRUCTURED OUTPUT:", type(structured_output))
   
            st.session_state.messages.append(
                {"role": "assistant", "content": "Here is the content i extracted from your image 🖼️: \n\n"+ str(structured_output)}
            )
            st.session_state["uploaded_file"] = True

        st.toast("Image analysis successfull !", icon="🎉")


    elif extension in [".pdf"]:
        save_uploaded_pdf(uploaded_file)
        pdf_loader = StructuredPDFOcerizer()
        with st.spinner("Performing OCR on the PDF..."):
            doc_pdf = pdf_loader.extract_text(
                pdf_path="temp/" + str(uploaded_file.name)
            )



        st.toast("OCR process successful!", icon="🎉")

        # Display the extracted text from the PDF
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Here is the extracted text from the given PDF 📄: \n\n"
                + doc_pdf,
            }
        )
        st.session_state["uploaded_file"] = True


@st.fragment
def print_suggestions(list_suggestions):
    selected = stp.pills(
        label="Suggestions", options=list_suggestions, index=None
    )
    if selected:
        st.write("You selected:", selected)
        process_query_v2(selected, st.session_state["streamlit_config"])

    return selected


def query_suggestion(query):
    """
    Provides suggestions for a given query by contextualizing it, breaking it down into subqueries,
    and printing the suggestions.

    Args:
        query (str): The query to provide suggestions for.

    Returns:
        selected: The selected suggestion.

    """
    # We first try to contextualize the query
    # kg = st.session_state["knowledge_graph"]
    # with st.spinner("Contextualizing the query..."):
    #     contextualized_query = kg.contextualize_query(query)

    context = "The query's answer is in the database of a company named Euranova which is specialized in AI and data science and R&D. \
        This company is located in France and Belgium, has several client and works on projects such as medical imaging, autonomous driving, and natural language processing."

    with st.spinner("Breaking down the query into subqueries..."):
        # We first break the query into subqueries
        breaker = QueryBreaker()
        list_sub_queries = breaker.break_query(query=query, context=context)[
            0:3
        ]  
        
    # We print the suggestions
    selected = print_suggestions(list_sub_queries)

    return selected


def create_transcription_pdf(transcription):
    """
    Create a PDF file containing the given transcription.

    Args:
        transcription (str): The text transcription to be included in the PDF.

    Returns:
        str: The file path of the created PDF.

    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    wrapped_text = textwrap.wrap(transcription, width=70)
    for line in wrapped_text:
        pdf.cell(200, 10, txt=line, ln=True)
    pdf_file_path = "temp/transcription.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path


def create_transcription_txt(transcription):
    """
    Create a text file containing the given transcription.

    Args:
        transcription (str): The transcription to be written to the text file.

    Returns:
        str: The file path of the created text file.
    """
    wrapped_text = textwrap.wrap(transcription, width=70)
    txt_file_path = "temp/transcription.txt"
    with open(txt_file_path, "w") as f:
        for line in wrapped_text:
            f.write(line + "\n")
    return txt_file_path


# def display_chat_history():
#     for message in st.session_state.messages:
#         if isinstance(message["content"], str):
#             st.chat_message(message["role"]).write(message["content"])
#         elif isinstance(message["content"], dict):
#             st.chat_message(message["role"]).json(message["content"])

def display_chat_history():
    for message in st.session_state.messages:
        if isinstance(message["content"], str):
            if message["role"] == "assistant" and "<output>" in message["content"]:
                # Split the content at the <output> tag
                reasoning, final_output = message["content"].split("<output>", 1)
                
                # Display the final output
                with st.chat_message(message["role"]):
                    st.write(final_output.strip())
                    
                    # Add an expander for the reasoning
                    with st.expander("Show intermediate reasoning steps"):
                        st.write(reasoning.strip())
            else:
                # For all other messages, display normally
                st.chat_message(message["role"]).write(message["content"])
        elif isinstance(message["content"], dict):
            st.chat_message(message["role"]).json(message["content"])


def clear_chat_history():
    st.session_state.messages = []
    st.toast("Chat history cleared!", icon="🧹")


def process_query_v2(query, streamlit_config):
    """
    Process the user's query and generate an answer using a configuration dictionary and call the RAG_answer function.

    Args:
        query (str): The user's query.
        config (dict): Configuration parameters.

    Returns:
        None
    """
    start_time = time.time()
    st.session_state.messages.append({"role": "user", "content": query})
    default_config = st.session_state["config"]
    
    #make a fusion between the default config and the streamlit config (the streamlit config has the priority)
    config = {**default_config, **streamlit_config}
    
    #we update the chat history to provide the LLM with the latest chat history
    config["chat_history"] = str("## Chat History: \n\n "+str(st.session_state['messages'])) #we keep track of the chat history
    
    if config["deep_search"]:
        st.toast("Deep search enabled ! it may take a while...", icon="⏳")
    
    st.chat_message("user").write(query)

    nb_tokens = token_calculation_prompt(str(query))

    if config["field_filter"] != []:
        enable_source_filter = True
        config["enable_source_filter"] = enable_source_filter


    if (
        len(config["field_filter"])
        == len(config["data_sources"])
    ):  # in this case we want to disable the source filter because we want to search in all sources available
        #print("Disabling source filter")
        enable_source_filter = False
        field_filter = []
        config["field_filter"] = field_filter

        config["enable_source_filter"] = enable_source_filter

    # we return the chunks to be able to display the sources
    config["return_chunks"] = True
    config["stream"] = True
    
    with st.spinner("Searching relevant documents and formulating answer..."):
        if config["deep_search"]:
            
            answer,sources = advanced_RAG_answer(
                query,
                default_config=default_config,
                config=config,
            )
            docs = []
            
        else:
            answer, docs, sources = RAG_answer(
                query,
                default_config=default_config,
                config=config,
            )


        with st.chat_message("assistant"):
            answer = st.write_stream(answer)

        end_time = time.time()
        st.toast(
            "Query answered in {:.2f} seconds!".format(end_time - start_time),
            icon="⏳",
        )
        
        # Add the answer to the chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Add the sources list to a session state variable
        st.session_state["sources"] = sources

        hallucination_scores = []
        # hallucination_scores = compute_hallucination_scores(answer, docs)

        display_sources_v2(
            sources, hallucination_scores
        )  # we do not compute the hallucination scores here its too long


def display_sources(sources, hallucination_scores):
    # Create a dictionary to hold the total hallucination score for each source
    source_hallucination_scores = {}
    for source, score in zip(sources, hallucination_scores):
        if source not in source_hallucination_scores:
            source_hallucination_scores[source] = score

    # Calculate the average hallucination score for each source
    source_counts = {source: sources.count(source) for source in sources}
    for source in source_hallucination_scores:
        # Convert the average hallucination score to percentage and format it without decimals and with a '%' sign
        source_hallucination_scores[source] = (
            f"{(source_hallucination_scores[source]*10):.0f}%"
        )

    # Combine the counts and hallucination scores into a single DataFrame
    data = []
    for source in sources:
        if source not in [
            d["Source (decreasing order of relevance)"] for d in data
        ]:  # Avoid adding duplicates
            data.append(
                {
                    "Source (decreasing order of relevance)": source,
                    "Number of times cited": source_counts[source],
                    "Average Hallucination Score (%)": source_hallucination_scores[
                        source
                    ],
                }
            )
    df_sources = pd.DataFrame(data)

    # Display the DataFrame
    st.dataframe(df_sources)


def display_sources_v2(sources, hallucination_scores):
    # Create a dictionary to count the occurrences of each source
    source_counts = {source: sources.count(source) for source in sources}

    # Combine the counts into a single DataFrame
    data = []
    for source in sources:
        if source not in [
            d["Source (decreasing order of relevance)"] for d in data
        ]:  # Avoid adding duplicates
            data.append(
                {
                    "Source (decreasing order of relevance)": source,
                    "Number of times cited": source_counts[source],
                }
            )
    df_sources = pd.DataFrame(data)

    # Display the DataFrame
    st.dataframe(df_sources)


@st.dialog("Feedback")
def feedback_dialog():
    feedback = st.selectbox(
        "How would you rate the answer?",
        [
            "Précise et Factuelle",
            "Information non trouvée/non existante",
            "Réponse totalement incorrecte",
            "Réponse partiellement incorrecte",
            "Réponse partielle à la question",
        ],
    )
    comment = st.text_input("Do you have any comment to make?")
    if st.button("Submit"):
        st.session_state.feedback = {"feedback": feedback, "comment": comment}
        st.toast("Feedback submitted! Thanks !", icon="😊")
        st.rerun()
