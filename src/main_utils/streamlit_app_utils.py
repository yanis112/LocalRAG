import concurrent.futures
import os
import time
from pathlib import Path
from threading import Lock

import qrcode
import streamlit as st
import yaml

from src.aux_utils.job_scrapper import JobAgent
from src.main_utils.generation_utils_v2 import LLM_answer_v3


def get_network_url():
    """
    Get the network URL of the local machine.

    This function attempts to determine the local machine's IP address by creating a temporary
    socket connection to an external server (Google's public DNS server at 8.8.8.8) and then
    retrieves the local IP address from the socket. It constructs a URL using this IP address
    and the default Streamlit port (8501).

    Returns:
        str: The network URL in the format "http://<local_ip>:8501". If an error occurs, it
        returns "http://localhost:8501".
    """
    import socket

    try:
        # Crée une connexion socket temporaire pour obtenir l'IP locale
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        port = 8501  # Port par défaut de Streamlit
        return f"http://{ip}:{port}"
    except Exception:
        return "http://localhost:8501"


def create_qrcode() -> bool:
    """
    Generate QR code from URL and save it to assets/qr_code.png

    Args:
        url: URL to encode in QR code
    Returns:
        bool: True if QR code was generated successfully
    """

    # obtain network URL
    url = get_network_url()
    # Check if already generated
    if st.session_state.get("is_qrcode_generated", False):
        return True

    try:
        # Create assets directory if it doesn't exist
        Path("assets").mkdir(exist_ok=True)

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(url)
        qr.make(fit=True)

        # Create and save image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        qr_image.save("assets/qr_code.png")

        # Update session state
        st.session_state["is_qrcode_generated"] = True
        return True

    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        return False


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
    from pydub import AudioSegment

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "input_file.wav")
    audio = AudioSegment.from_file(uploaded_file)
    audio.export(file_path, format="wav")
    return file_path


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

    if "messages" not in st.session_state:
        st.session_state.messages = []


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

import asyncio
from pathlib import Path
import os

import asyncio
import concurrent.futures
import os
from pathlib import Path

import streamlit as st


import asyncio
import concurrent.futures
import os
from pathlib import Path

import streamlit as st


def handle_multiple_uploaded_files(uploaded_files, parallel: bool = False):
    """
    Traite plusieurs fichiers uploadés avec option de traitement en parallèle (async) ou séquentiel.
    
    Args:
        uploaded_files: Liste des fichiers uploadés à traiter.
        parallel: Si True, traite les fichiers en parallèle (en utilisant asyncio).
                  Sinon, les traite séquentiellement.
    """
    total_files = len(uploaded_files)

    async def process_files():
        if parallel:
            # Crée une tâche async pour chaque fichier et attend leur exécution en parallèle.
            tasks = [
                handle_single_file(file)
                for file in uploaded_files
            ]
            await asyncio.gather(*tasks)
        else:
            # Traitement séquentiel : on attend le traitement de chaque fichier l'un après l'autre.
            for idx, file in enumerate(uploaded_files, 1):
                with st.spinner(f"Processing file {idx}/{total_files}: {file.name}",show_time=True):
                    await handle_single_file(file)
    
    asyncio.run(process_files())
    # On affiche l'historique de chat mis à jour (une seule fois et à la fin sinon on risque de le rafraîchir plusieurs !)
    display_chat_history()







@st.fragment
async def handle_single_file(uploaded_file):
    """
    Traite le fichier uploadé et l'analyse en fonction de son extension.
    
    Pour :
    - Les fichiers audio (.mp3, .wav, .m4a, .mp4) :
         - Sauvegarde le fichier audio en format wav.
         - Transcrit l'audio en utilisant YouTubeTranscriber.
         - Crée un fichier texte de transcription.
    - Les fichiers image (.png, .jpg, .jpeg) :
         - Affiche l'image.
         - Sauvegarde l'image dans le dossier temporaire.
         - Analyse l'image via UniversalImageLoader.
    - Les fichiers PDF (.pdf) :
         - Sauvegarde le PDF.
         - Exécute l'OCR sur le PDF avec StructuredPDFOcerizer.
    
    La fonction met à jour l'état de l'application (st.session_state) avec les résultats.
    """
    temp_dir = "temp"
    extension = str(Path(uploaded_file.name).suffix).lower()

    if extension in [".mp3", ".wav", ".m4a", ".mp4"]:
        from src.aux_utils.transcription_utils_v3 import YouTubeTranscriber
        file_path = save_audio_as_wav(uploaded_file, temp_dir)
        with st.spinner("Transcribing audio 🎤 ...",show_time=True):
            # On exécute ici la transcription de manière asynchrone.
            yt = YouTubeTranscriber(chunk_size=st.session_state["config"]["transcription_chunk_size"], batch_size=1)
            # On suppose que yt.transcribe peut être awaité ou, sinon, vous pouvez utiliser asyncio.to_thread(...)
            transcription = await asyncio.to_thread(yt.transcribe, file_path, method="groq", diarization=st.session_state["diarization_enabled"])
        
        st.toast("Transcription successful !", icon="🎤")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Here is your transcribed audio 🔉📜 ! \n\n" + str(transcription),
            }
        )
        st.session_state["uploaded_file"] = True
        st.session_state["audio_transcription"] = transcription
        st.session_state["external_resources_list"].append(transcription)

    elif extension in [".png", ".jpg", ".jpeg"]:
        st.image(uploaded_file)
        uploaded_file.seek(0)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
        from src.aux_utils.vision_utils_v2 import ImageAnalyzerAgent
        with st.spinner("Analyzing the image... it should take less than 2 minutes 😜",show_time=True):
            analyser = ImageAnalyzerAgent()
            with open("prompts/image2markdown.txt", "r", encoding="utf-8") as f:
                prompt = f.read()
            # Supposons que analyser.describe soit synchrone : on le lance dans un thread.
            output = await asyncio.to_thread(
                analyser.describe,
                image_path=os.path.join(temp_dir, uploaded_file.name),
                prompt=prompt,
                vllm_provider=st.session_state["config"]["vllm_provider"],
                vllm_name=st.session_state["config"]["vllm_model_name"],
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Here is the content I extracted from your image 🖼️: \n\n" + str(output),
                }
            )
            st.session_state["uploaded_file"] = True
            st.session_state["external_resources_list"].append(output)
        st.toast("Image analysis successful !", icon="🎉")

    elif extension in [".pdf"]:
        from src.main_utils.utils import StructuredPDFOcerizer
        save_uploaded_pdf(uploaded_file)
        pdf_loader = StructuredPDFOcerizer()
        with st.spinner("Performing OCR on the PDF...",show_time=True):
            # Lance le traitement OCR dans un thread pour éviter le blocage
            doc_pdf = await asyncio.to_thread(pdf_loader.extract_text, pdf_path=os.path.join(temp_dir, str(uploaded_file.name)))
        st.toast("OCR process successful!", icon="🎉")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Here is the extracted text from the given PDF 📄: \n\n" + doc_pdf,
            }
        )
        st.session_state["uploaded_file"] = True
        st.session_state["external_resources_list"].append(doc_pdf)
    
    

# Lock for thread-safe progress updates
# spinner_lock = Lock()


# def handle_multiple_uploaded_files(uploaded_files, parallel: bool = False):
#     """
#     Handles multiple uploaded files with optional parallel processing.

#     Args:
#         uploaded_files: List of uploaded files to process
#         parallel: If True, process files in parallel using threading. If False, process sequentially.
#     """
#     total_files = len(uploaded_files)

#     def thread_safe_handle_file(file):
#         with spinner_lock:
#             with st.spinner(f"Processing: {file.name}"):
#                 return handle_single_file(file)

#     if parallel:
#         # Determine optimal number of workers
#         max_workers = min(total_files, 4)  # Limit concurrent threads
#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all files for processing
#             futures = [
#                 executor.submit(thread_safe_handle_file, file)
#                 for file in uploaded_files
#             ]
#             # Wait for all to complete
#             concurrent.futures.wait(futures)

#     else:
#         # Sequential processing
#         for idx, file in enumerate(uploaded_files, 1):
#             with st.spinner(f"Processing file {idx}/{total_files}: {file.name}"):
#                 handle_single_file(file)


# @st.fragment
# def handle_single_file(uploaded_file):
#     """
#     Handles the uploaded file and processes it based on its extension.
#     Parameters:
#     uploaded_file (UploadedFile): The file uploaded by the user.
#     Returns:
#     bytes: The data of the processed file if applicable.
#     Processes:
#     - For audio files (mp3, wav, m4a, mp4):
#         - Saves the audio file as a wav file.
#         - Transcribes the audio using YouTubeTranscriber.
#         - Creates a transcription text file.
#         - Displays success messages and returns the transcription data.
#     - For image files (png, jpg, jpeg):
#         - Displays the image.
#         - Saves the image in the temp folder.
#         - Analyzes the image using UniversalImageLoader.
#         - Displays success messages and the extracted content from the image.
#     - For PDF files (pdf):
#         - Saves the uploaded PDF.
#         - Performs OCR on the PDF using StructuredPDFOcerizer.
#         - Displays success messages and the extracted text from the PDF.
#     """

#     temp_dir = "temp"
#     extension = str(Path(uploaded_file.name).suffix)

#     if extension in [".mp3", ".wav", ".m4a", ".mp4"]:
#         from src.aux_utils.transcription_utils_v3 import (
#             YouTubeTranscriber,  # try v0 if it doesnt work
#         )

#         file_path = save_audio_as_wav(uploaded_file, temp_dir)
#         # st.success(f"File saved as {file_path}")

#         with st.spinner("Transcribing audio 🎤 ..."):
#             yt = YouTubeTranscriber(chunk_size=st.session_state["config"]["transcription_chunk_size"], batch_size=1)
#             transcription = yt.transcribe(
#                 file_path,
#                 method="groq",
#                 diarization=st.session_state["diarization_enabled"],
#             )

#         # print("TRANSCRIPTION: ", transcription)
#         st.toast("Transcription successful !", icon="🎤")

#         st.session_state.messages.append(
#             {
#                 "role": "assistant",
#                 "content": "Here is your transcribed audio 🔉📜 ! \n\n"
#                 + str(transcription),
#             }
#         )

#         st.session_state["uploaded_file"] = True
#         st.session_state["audio_transcription"] = transcription
#         st.session_state["external_resources_list"].append(transcription)

#         # txt = create_transcription_txt(transcription)
#         # print("TXT FILE CREATED !")

#         # with open(txt, "rb") as f:
#         #     data = f.read()

#         # st.toast("Transcription successful!", icon="🎉")
#         # print("DOWNLOAD BUTTON CREATED !")

#         # return data

#     elif extension in [".png", ".jpg", ".jpeg"]:
#         st.image(uploaded_file)
#         # save the image in the temp folder
#         uploaded_file.seek(0)

#         # create the temp folder if not existing yet
#         if not os.path.exists("temp"):
#             os.makedirs("temp")

#         with open("temp/" + uploaded_file.name, "wb") as f:
#             f.write(uploaded_file.read())

#         from src.aux_utils.vision_utils_v2 import ImageAnalyzerAgent

#         with st.spinner("Analyzing the image... it should take less than 2 minutes 😜"):
#             # load the universal image loader
#             analyser = ImageAnalyzerAgent()
#             # load the task prompt from prompts/image2markdown.txt
#             with open("prompts/image2markdown.txt", "r", encoding="utf-8") as f:
#                 prompt = f.read()

#             output = analyser.describe(
#                 image_path="temp/" + uploaded_file.name,
#                 prompt=prompt,
#                 vllm_provider=st.session_state["config"]["vllm_provider"],
#                 vllm_name=st.session_state["config"]["vllm_model_name"],
#             )
#             print("IMAGE ANALYSIS OUTPUT: ", output)

#             st.session_state.messages.append(
#                 {
#                     "role": "assistant",
#                     "content": "Here is the content i extracted from your image 🖼️: \n\n"
#                     + str(output),
#                 }
#             )
#             st.session_state["uploaded_file"] = True

#             # append the external resources obtained to the session state 'external_resources_list'
#             st.session_state["external_resources_list"].append(output)

#         st.toast("Image analysis successfull !", icon="🎉")

#     elif extension in [".pdf"]:
#         from src.main_utils.utils import StructuredPDFOcerizer

#         save_uploaded_pdf(uploaded_file)

#         pdf_loader = StructuredPDFOcerizer()
#         with st.spinner("Performing OCR on the PDF..."):
#             doc_pdf = pdf_loader.extract_text(
#                 pdf_path="temp/" + str(uploaded_file.name)
#             )

#         st.toast("OCR process successful!", icon="🎉")

#         # Display the extracted text from the PDF
#         st.session_state.messages.append(
#             {
#                 "role": "assistant",
#                 "content": "Here is the extracted text from the given PDF 📄: \n\n"
#                 + doc_pdf,
#             }
#         )
#         st.session_state["uploaded_file"] = True
#         # append the external resources obtained to the session state 'external_resources_list'
#         st.session_state["external_resources_list"].append(doc_pdf)

#     # refresh the displaying of chat messages
#     display_chat_history()


@st.fragment
def print_suggestions(list_suggestions):
    from streamlit_pills import stp

    selected = stp.pills(label="Suggestions", options=list_suggestions, index=None)
    if selected:
        st.write("You selected:", selected)
        process_query(
            selected,
            st.session_state["streamlit_config"],
            st.session_state["rag_agent"],
        )

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
    from src.main_utils.agentic_rag_utils import QueryBreaker

    context = "The query's answer is in the database of a company named Euranova which is specialized in AI and data science and R&D. \
        This company is located in France and Belgium, has several client and works on projects such as medical imaging, autonomous driving, and natural language processing."

    with st.spinner("Breaking down the query into subqueries...",show_time=True):
        # We first break the query into subqueries
        breaker = QueryBreaker()
        list_sub_queries = breaker.break_query(query=query, context=context)[0:3]

    # We print the suggestions
    selected = print_suggestions(list_sub_queries)

    return selected


def display_chat_history():
    """
    Display the chat history stored in the Streamlit session state.

    This function iterates through the messages in the session state and displays
    them in the Streamlit app. It handles messages differently based on their content
    and role:
    
    - If the message is from the assistant and contains an "<output>" tag, it splits
      the content at the tag, displays the final output, and provides an expander to
      show the intermediate reasoning steps.
    - For other string messages, it displays them normally.
    - If the message content is a dictionary, it displays the content as JSON.

    The function uses custom avatars for the chat messages based on the role of the
    sender.

    Raises:
        AttributeError: If `st.session_state.messages` is not defined.
    """
    
    #test for the existance of this variable
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            if isinstance(message["content"], str):
                if message["role"] == "assistant" and "</think>" in message["content"]:
                    # Split the content at the <output> tag
                    reasoning, final_output = message["content"].split("</think>", 1)

                    # Display the final output
                    with st.chat_message(message["role"], avatar="assets/icon_ai_2.jpg"):
                        st.write(final_output.strip())

                        # Add an expander for the reasoning
                        with st.expander("Show intermediate reasoning steps"):
                            st.write(reasoning.strip())
                else:
                    # For all other messages, display normally
                    #with st.session_state["chat_container"]: #NOUVEAU
                    st.chat_message(message["role"], avatar="assets/icon_ai_2.jpg").write(
                        message["content"]
                    )
            elif isinstance(message["content"], dict):
                #with st.session_state["chat_container"]: #NOUVEAU
                st.chat_message(message["role"], avatar="assets/icon_ai_2.jpg").json(
                    message["content"]
                )
                
                
def export_to_notion(text):
    """
    This function when called takes the last message in session state and sends it for it to be converted to a 
    notion page with appropriate name
    """
    from src.aux_utils.notion_utils import NotionAgent
    from src.main_utils.generation_utils_v2 import LLM_answer_v3
    from langchain_core.prompts import PromptTemplate
    notion_agent= NotionAgent()

    # load the meeting summarization prompt from prompts/meeting_summary_prompt.txt
    with open("prompts/text2title.txt", "r", encoding="utf-8") as f:
        template = f.read()

    template = PromptTemplate.from_template(template)
    full_prompt = template.format(
        text=text
    )
    # give it to the LLM
    with st.spinner("Generating title for Notion page...",show_time=True):
        title = LLM_answer_v3(
            prompt=full_prompt,
            stream=False,
            model_name="llama-3.3-70b-versatile",
            llm_provider="groq",
            temperature=st.session_state["streamlit_config"]["temperature"],
        )
        

    
    page_id = notion_agent.create_page_from_markdown(
    text,
    page_title=title
    )
    st.toast(f"Notion page created with name '{title}'!", icon="📋")
    


def clear_chat_history():
    st.session_state.messages = []
    st.toast("Chat history cleared!", icon="🧹")
    # rerun the app to clear the chat
    st.rerun()


def process_audio_recording(audio):
    """
    Process an audio recording by saving it to a temporary file and transcribing it.
    This function saves the provided audio recording to a temporary file named
    'recorded_audio.wav' in a 'temp' directory. It then uses the YouTubeTranscriber
    class to transcribe the audio file and displays the transcription in a Streamlit
    chat message and a toast notification.
    Args:
        audio: An audio recording object that supports the `getbuffer` method to
               retrieve the audio data.
    Returns:
        str: The transcription of the audio recording.
    """
    from src.aux_utils.transcription_utils_v3 import YouTubeTranscriber

    if not os.path.exists("temp"):
        os.makedirs("temp")

    with open("temp/recorded_audio.wav", "wb") as f:
        f.write(audio.getbuffer())

    print("Audio saved as recorded_audio.wav")

    with st.spinner("Transcribing audio...🎤",show_time=True):
        yt = YouTubeTranscriber(chunk_size=st.session_state["config"]["transcription_chunk_size"])
        transcription = yt.transcribe("temp/recorded_audio.wav", method="groq",diarization=st.session_state["diarization_enabled"])

    print("TRANSCRIPTION: ", transcription)

    st.chat_message("assistant").write(
        "🎤 Audio recorded and transcribed: \n\n" + transcription
    )
    st.toast("Transcription successful!", icon="🎉")

    # add transcription to the session state
    st.session_state["audio_transcription"] = transcription

    return transcription


def process_query(query, streamlit_config, rag_agent):
    """
    Process the user's query and generate an answer using a configuration dictionary and call the RAG_answer function.

    Args:
        query (str): The user's query.
        config (dict): Configuration parameters.
        rag_agent (RAGAgent): An instance of the RAGAgent class.

    Returns:
        None
    """
    from src.aux_utils.text_classification_utils import IntentClassifier

    start_time = time.time()
    st.session_state.messages.append({"role": "user", "content": query})
    default_config = st.session_state["config"]

    # make a fusion between the default config and the streamlit config (the streamlit config has the priority)
    config = {**default_config, **streamlit_config}

    # add this config to the rag agent
    rag_agent.merged_config = config

    # we load the intent classifier into a session state variable so that we check if the intent is already loaded

    if "intent_classifier" not in st.session_state:
        st.session_state["intent_classifier"] = IntentClassifier(config=config)

    # we update the chat history to provide the LLM with the latest chat history
    config["chat_history"] = str(
        "## Chat History: \n\n " + str(st.session_state["messages"])
    )  # we keep track of the chat history

    if config["deep_search"]:  # if deep search is enabled we notify the user
        st.toast("Deep search enabled ! it may take a while...", icon="⏳")

    # we write the query in the chat
    st.chat_message("user", avatar="assets/icon_human.jpg").write(query)

    if (
        config["field_filter"] != []
    ):  # if no specific field is selected we disable the source filter
        config["enable_source_filter"] = True
    else:
        config["enable_source_filter"] = False

    # we return the chunks to be able to display the sources
    config["return_chunks"] = True
    config["stream"] = True

    if config[
        "deep_search"
    ]:  # if deep search is enabled we use the advanced_RAG_answer function and no intent classifier !
        with st.spinner("Searching relevant documents and formulating answer...",show_time=True):
            answer, sources = rag_agent.advanced_RAG_answer(query)
            docs = []

    else:
        from src.main_utils.utils import extract_url

        # we check if the query contains the command @add to add a document to the vectorstore
        if "@add" in query:
            # we create a clean query without the @add command
            if extract_url(query)!=None:
                query = extract_url(query)
            # we import the external knowledge manager

            if "http" in query:
                # we extract the rescource from the link
                from src.main_utils.link_gestion import ExternalKnowledgeManager

                link_manager = ExternalKnowledgeManager(config, client=rag_agent.client)
                link_manager.extract_rescource_from_link(query.replace("@add", ""))
                link_manager.index_rescource()
                # we stop the process here
                st.toast("New rescource indexed !", icon="🎉")
                return None
            else:
                # we extract directly the pasted rescource
                st.toast(
                    "No url detected, processing the pasted rescource...", icon="🔍"
                )
                from src.main_utils.link_gestion import ExternalKnowledgeManager
                link_manager = ExternalKnowledgeManager(config, client=rag_agent.client)
                link_manager.extract_rescource(query) #extract raw text resource
                link_manager.index_rescource()
                
                pass

        with st.spinner("Determining query intent 🧠 ...",show_time=True):
            # we detect the intent of the query
            intent = st.session_state["intent_classifier"].classify(query, method="LLM")
            st.toast("Intent detected: " + intent, icon="🧠")

        if intent == "employer contact writing":
            from src.aux_utils.job_agent_v2 import JobWriterAgent

            job_agent = JobWriterAgent(config=config)

            answer, _ = job_agent.generate_content(query)
            sources = []
            docs = []

        elif intent == "email support inquiry":
            # fetch the last 100 emails and search for the answer
            from src.aux_utils.email_utils import EmailAgent

            with st.spinner("Fetching new emails...",show_time=True):
                email_utils = EmailAgent()
                # email_utils.connect()
                email_utils.fetch_new_emails(last_k=100)
                # email_utils.disconnect()

            # fill the vectorstore withg the new emails
            from src.main_utils.vectorstore_utils_v4 import VectorAgent

            with st.spinner("Filling the vectorstore with new emails...",show_time=True):
                agent = VectorAgent(
                    default_config=config, qdrant_client=rag_agent.client
                )
                agent.fill()
                print("Vectorstore filled with new emails !")

            config["data_sources"] = ["emails"]
            config["enable_source_filter"] = True
            config["field_filter"] = ["emails"]
            # actualize the rag agent config to the new config
            rag_agent.merged_config = config
            with st.spinner(
                "Searching relevant documents and formulating answer 📄 ...",show_time=True
            ):
                answer, docs, sources = rag_agent.RAG_answer(query)

        elif intent == "job search assistance":
            print("Launched job search !")

            # from src.aux_utils.job_scrapper import JobAgent

            config["data_sources"] = ["jobs"]
            config["enable_source_filter"] = True
            config["field_filter"] = ["jobs"]

            prompt = f'''Here is a textual query for a job search from a user "{query}", 
            please provide the structured search parameters in the following dictionary format:
            {{"search_terms": ["keywords_1", "keywords_2", ...], "locations": ["location_1", "location_2", ...]}}. Return the
            the str dict without preamble.'''

            answer = LLM_answer_v3(
                prompt,
                model_name=config["model_name"],
                llm_provider=config["llm_provider"],
                stream=False,
                temperature=st.session_state["streamlit_config"]["temperature"],
            )
            # transform the str dict into real dict
            import ast

            def str_to_dict(dict_str: str) -> dict:
                """
                Convert a string representation of a dictionary to an actual dictionary.

                Args:
                    dict_str (str): String representation of dictionary

                Returns:
                    dict: Converted dictionary

                Raises:
                    ValueError: If string cannot be converted to dictionary
                """
                try:
                    # Remove any whitespace and normalize quotes
                    cleaned_str = dict_str.strip().replace("'", '"')
                    return ast.literal_eval(cleaned_str)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid dictionary string: {e}")

            print("Raw dict answer:", answer)
            dict_params = str_to_dict(answer)
            print("Obtained dict parameters:", dict_params)
            # get search terms from the dict
            search_terms = dict_params["search_terms"]
            locations = dict_params["locations"]

            # scrapping jobs
            with st.spinner("Scraping job offers...",show_time=True):
                try:
                    print("Initializing job scrapper...")
                    job_scrapper = JobAgent(is_remote=False)

                    job_scrapper.scrape_and_convert(
                        search_terms=search_terms,
                        locations=locations,
                        hours_old=200,
                        results_wanted=20,
                        # add locations to each of the search_terms as google_search terms
                        google_search_terms=[
                            search_term + " " + location
                            for search_term in search_terms
                            for location in locations
                        ],
                        is_remote=False,
                    )
                except Exception as e:
                    print("EXCEPTION IN JOB SCRAPPING:", e)
                    st.error("An error occured while scrapping the job offers !")

            # fill the vectorstore with the new job offers
            from src.main_utils.vectorstore_utils_v4 import VectorAgent

            with st.spinner("Filling the vectorstore with new job offers...",show_time=True):
                agent = VectorAgent(
                    default_config=config, qdrant_client=rag_agent.client
                )
                agent.fill()
                print("Vectorstore filled with new job offers !")

            with st.spinner("Searching relevant jobs and formulating answer 📄 ...",show_time=True):
                answer, docs, sources = rag_agent.RAG_answer(query)

        # elif intent == "social media content creation":
        #     from src.aux_utils.auto_instagram_publi import (
        #         instagram_descr_prompt,
        #     )

        #     with st.spinner("Generating a text for an instagram post..."):
        #         answer = LLM_answer_v3(
        #             prompt=instagram_descr_prompt(query),
        #             stream=True,
        #             model_name=config["model_name"],
        #             llm_provider=config["llm_provider"],
        #         )
        #         sources = []

        elif intent == "graph creation request":
            from src.aux_utils.graph_maker_utils import GraphMaker

            with st.spinner("Generating the graph..."):
                graph_maker = GraphMaker(
                    model_name=st.session_state["config"]["model_name"],
                    llm_provider=st.session_state["config"]["llm_provider"],
                )
                output_path = graph_maker.generate_graph(base_prompt=query)
                # show the svg file in the streamlit app
                st.image(output_path)
                sources = []
                docs = []
                answer = "Here is the generated graph !"
                # convert answer to a generator like object to be treated as a stream
                answer = (line for line in [answer])

        elif intent == "meeting notes summarization":
            from langchain_core.prompts import PromptTemplate
            from src.aux_utils.notion_utils import NotionAgent

            # load the meeting summarization prompt from prompts/meeting_summary_prompt.txt
            with open("prompts/meeting_summary_prompt.txt", "r", encoding="utf-8") as f:
                template = f.read()

            template = PromptTemplate.from_template(template)
            full_prompt = template.format(
                transcription=st.session_state["audio_transcription"], user_query=query
            )

            # give it to the LLM
            with st.spinner("Generating the meeting summary...",show_time=True):
                raw_answer = LLM_answer_v3(
                    prompt=full_prompt,
                    stream=False,
                    model_name=config["model_name"],
                    llm_provider=config["llm_provider"],
                    temperature=st.session_state["streamlit_config"]["temperature"],
                )
                sources = []
                docs = []
                # transform the answer into a stream / generator object
                
                #(line for line in raw_answer.split("\n"))
                # save the transcription in a file, the name of the file is the date of the meeting (today)

                with open(
                    "data/meeting_summary/" + str(time.strftime("%Y-%m-%d")) + ".txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    # write a header with the date at the beginning of the file
                    f.write(
                        "Meeting summary for the date: "
                        + str(time.strftime("%Y-%m-%d"))
                        + "\n\n"
                    )
                    f.write(raw_answer)
                    
                #create a notion page with the meeting summary
                notion_agent = NotionAgent()
                
                with st.spinner("Creating a Notion page for transcription...",show_time=True):
                    notion_agent.create_page_from_markdown(
                    markdown_content=raw_answer,
                    #for page title we take first line of raw_answer
                    page_title = raw_answer.split('\n')[0]
                    )
                    st.toast("Notion page created !", icon="📋")
                    
                    

                # add a toast to notify the user that the meeting summary has been saved
                st.toast("Meeting summary saved !", icon="🎉")
                # fill the vectorstore with the new meeting summary
                from src.main_utils.vectorstore_utils_v4 import VectorAgent

                vector_agent = VectorAgent(
                    default_config=st.session_state["config"],
                    qdrant_client=rag_agent.client,
                )
                vector_agent.fill()
                print("Vectorstore filled with new meeting summary !")
                st.toast("Meeting summary indexed !", icon="📋")
                # put st.session_state["audio_transcription"] to None now that the meeting summary has been saved
                del st.session_state["audio_transcription"]
                answer = "📋 Meeting summary saved in your Notion workspace and in the database !"
                answer= (line for line in answer.split("\n"))

        elif intent == "sheet or table info extraction":
            config["data_sources"] = ["sheets"]
            config["enable_source_filter"] = True
            config["field_filter"] = ["sheets"]
            # put back config to the rag agent
            rag_agent.merged_config = config
            # initialize the vector agent
            from src.main_utils.vectorstore_utils_v4 import VectorAgent

            vector_agent = VectorAgent(
                default_config=st.session_state["config"],
                qdrant_client=rag_agent.client,
            )
            # remove all the current files of data/sheets
            vector_agent.delete(folders=["data/sheets"])
            # fetch all sheets using sheet agent
            from src.aux_utils.google_sheets_agent import GoogleSheetsAgent

            sheet_agent = GoogleSheetsAgent(
                credentials_path="google_json_key/python-sheets-446015-aa8eef72c872.json",
                save_path="data/sheets",
                temp_path="temp",
            )
            with st.spinner("Fetching the sheets..."):
                sheet_agent.fetch_and_save(spreadsheet_name="RechercheEmploi")
            # fill the vectorstore with the new sheets
            vector_agent.fill()
            # answer the query
            with st.spinner(
                "Searching relevant documents and formulating answer 📄 ..."
            ):
                answer, docs, sources = rag_agent.RAG_answer(query)

        elif intent == "prompt engineering request":
            config["data_sources"] = ["prompts"]
            config["enable_source_filter"] = True
            config["field_filter"] = ["prompts"]

            from src.aux_utils.cinematic_agent_prompter import AgentCinematicExpert

            with st.spinner("🧠 Refining your prompt..."):
                print("model currently used: ", config["model_name"])
                agent = AgentCinematicExpert(
                    model_name=config["model_name"], llm_provider=config["llm_provider"]
                )
                answer = agent.transform_chain(query)
                sources = []
                docs = []
                # transform the answer into a stream / generator object
                answer = (line for line in answer.split("\n"))

        # elif intent =="previous answer correction":
        #     from langchain_core.prompts import PromptTemplate
        #     #load the prompt for previous answer correction
        #     with open("prompts/previous_answer_correction.txt", "r", encoding='utf-8') as f:
        #         template = f.read()

        #     template= PromptTemplate.from_template(template)

        #     #convert the history of messages to a dictionary str
        #     template.format(historique=str(st.session_state["messages"]),user_query=query)

        #     print("################## TEMPLATE ##################")
        #     print("FULL ANSWER CORRECTION PROMPT: ", template)
        #     print("################## TEMPLATE ##################")

        #     # give it to the LLM
        #     with st.spinner("Generating the corrected answer..."):
        #         answer = LLM_answer_v3(
        #             prompt=template,
        #             stream=True,
        #             model_name=config["model_name"],
        #             llm_provider=config["llm_provider"],
        #         )
        #         sources = []

        else:  # normal search
            with st.spinner(
                "Searching relevant documents and formulating answer 📄 ..."
            ):
                answer, docs, sources = rag_agent.RAG_answer(query)

        with st.chat_message("assistant", avatar="assets/icon_ai_2.jpg"):
            answer = st.write_stream(answer)

        end_time = time.time()
        st.toast(
            "Query answered in {:.2f} seconds!".format(end_time - start_time),
            icon="⏳",
        )

        # Add the answer to the chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Add the sources list to a session state variable
        st.session_state["sources"] = sources
        
        
        display_sources(sources, docs)
            


# @st.fragment
# def display_sources(sources, docs=None):
#     """
#     Display sources and store chunks in session_state
#     Args:
#         sources (list): List of source file paths.
#         docs (list): List of langchain documents objects: Document(metadata={...},page_content="...").
#     """
#     import os
#     import webbrowser
#     from streamlit_extras.stylable_container import stylable_container

#     if docs and len(docs) != len(sources):
#         st.toast("Erreur de correspondance sources/chunks", icon="⚠️")
#         return

#     subdomain_emojis = {
#         "jobs": "💼",
#         "politique": "🏛️",
#         "internet": "🌐",
#         "emails": "📧",
#         "prompts": "📝",
#     }

#     sources_dict = {}
#     for idx, source in enumerate(sources):
#         absolute_path = os.path.abspath(source)
#         filename = os.path.basename(absolute_path)

#         subdomain = next(
#             (domain for domain in subdomain_emojis if domain in absolute_path.lower()),
#             None,
#         )
#         emoji = subdomain_emojis.get(subdomain, "📄")

#         entry = sources_dict.get(absolute_path, {
#             "Filename": filename,
#             "Count": 0,
#             "Emoji": emoji,
#             "Chunks": []
#         })
        
#         entry["Count"] += 1
#         if docs:
#             entry["Chunks"].append(docs[idx])
        
#         sources_dict[absolute_path] = entry

#     st.session_state.sources_dict = sources_dict

#     # Affichage UI
#     with st.expander("Sources 📑"):
#         with stylable_container(
#             key="sources_container",
#             css_styles="""
#             /* [CSS original conservé] */
#             """
#         ):
#             for i, (path, meta) in enumerate(sources_dict.items()):
#                 cols = st.columns([0.1, 0.5, 0.5, 0.1])
#                 with cols[0]: st.write(meta['Emoji'])
#                 with cols[1]: st.write(meta['Filename'])
#                 with cols[2]: st.write(f"({meta['Count']})")
#                 with cols[3]: 
#                     if st.button("🔍", key=f"button_{i}"):
#                         webbrowser.open(f"file://{path}", new=1)

import os
import streamlit as st
from streamlit import switch_page
from streamlit_extras.stylable_container import stylable_container

@st.fragment
def display_sources(sources, docs=None):
    """
    Display sources and store chunks in session_state
    Args:
        sources (list): List of source file paths.
        docs (list): List of langchain documents objects: Document(metadata={...},page_content="...").
    """

    if docs and len(docs) != len(sources):
        st.toast("Erreur de correspondance sources/chunks", icon="⚠️")
        return

    subdomain_emojis = {
        "jobs": "💼",
        "politique": "🏛️",
        "internet": "🌐",
        "emails": "📧",
        "prompts": "📝",
    }

    sources_dict = {}
    for idx, source in enumerate(sources):
        absolute_path = os.path.abspath(source)
        filename = os.path.basename(absolute_path)

        subdomain = next(
            (domain for domain in subdomain_emojis if domain in absolute_path.lower()),
            None,
        )
        emoji = subdomain_emojis.get(subdomain, "📄")

        entry = sources_dict.get(absolute_path, {
            "Filename": filename,
            "Count": 0,
            "Emoji": emoji,
            "Chunks": []
        })

        entry["Count"] += 1
        if docs:
            entry["Chunks"].append(docs[idx])

        sources_dict[absolute_path] = entry

    st.session_state.sources_dict = sources_dict

    # Affichage UI
    with st.expander("Sources 📑"):
        with stylable_container(
            key="sources_container",
            css_styles="""
            /* [CSS original conservé] */
            """
        ):
            for i, (path, meta) in enumerate(sources_dict.items()):
                cols = st.columns([0.1, 0.5, 0.5, 0.1])
                with cols[0]: st.write(meta['Emoji'])
                with cols[1]: st.write(meta['Filename'])
                with cols[2]: st.write(f"({meta['Count']})")
                with cols[3]:
                    # Utilisez l'ID du premier chunk comme ancre (si disponible)
                    first_chunk_id = meta['Chunks'][0].metadata.get('_id', '') if meta['Chunks'] else ''
                    if st.button("🔍", key=f"button_{i}"):
                        switch_page(f"pages/sources_page.py")

if __name__ == "__main__":
    pass
