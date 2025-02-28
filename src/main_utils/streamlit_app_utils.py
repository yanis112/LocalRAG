import os
import time
from pathlib import Path

import qrcode
import streamlit as st
import yaml
from streamlit import switch_page
from streamlit_extras.stylable_container import stylable_container

from src.aux_utils.job_scrapper import JobAgent
from src.aux_utils.email_utils import EmailAgent
from src.main_utils.generation_utils_v2 import LLM_answer_v3, RAGAgent
from src.main_utils.vectorstore_utils_v5 import VectorAgent


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
            st.write("Dont press enter key when entering inputs, it will submit the form !")
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
                    "data/meeting_transcriptions/" + st.session_state["document_title"] + ".txt",
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
            tasks = [handle_single_file(file) for file in uploaded_files]
            await asyncio.gather(*tasks)
        else:
            # Traitement séquentiel : on attend le traitement de chaque fichier l'un après l'autre.
            for idx, file in enumerate(uploaded_files, 1):
                with st.spinner(
                    f"Processing file {idx}/{total_files}: {file.name}", show_time=True
                ):
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
        with st.spinner("Transcribing audio 🎤 ...", show_time=True):
            # On exécute ici la transcription de manière asynchrone.
            yt = YouTubeTranscriber(
                chunk_size=st.session_state["config"]["transcription_chunk_size"], batch_size=1
            )
            # On suppose que yt.transcribe peut être awaité ou, sinon, vous pouvez utiliser asyncio.to_thread(...)
            transcription = await asyncio.to_thread(
                yt.transcribe,
                file_path,
                method="groq",
                diarization=st.session_state["diarization_enabled"],
            )

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

        with st.spinner(
            "Analyzing the image... it should take less than 2 minutes 😜", show_time=True
        ):
            analyser = ImageAnalyzerAgent(config=st.session_state["config"])    
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
                    "content": "Here is the content I extracted from your image 🖼️: \n\n"
                    + str(output),
                }
            )
            st.session_state["uploaded_file"] = True
            st.session_state["external_resources_list"].append(output)
        st.toast("Image analysis successful !", icon="🎉")

    elif extension in [".pdf"]:
        from src.main_utils.utils import StructuredPDFOcerizer

        save_uploaded_pdf(uploaded_file)
        pdf_loader = StructuredPDFOcerizer()
        with st.spinner("Performing OCR on the PDF...", show_time=True):
            # Lance le traitement OCR dans un thread pour éviter le blocage
            doc_pdf = await asyncio.to_thread(
                pdf_loader.extract_text, pdf_path=os.path.join(temp_dir, str(uploaded_file.name))
            )
        st.toast("OCR process successful!", icon="🎉")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Here is the extracted text from the given PDF 📄: \n\n" + doc_pdf,
            }
        )
        st.session_state["uploaded_file"] = True
        st.session_state["external_resources_list"].append(doc_pdf)


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

    with st.spinner("Breaking down the query into subqueries...", show_time=True):
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

    # test for the existance of this variable
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
                    # with st.session_state["chat_container"]: #NOUVEAU
                    st.chat_message(message["role"], avatar="assets/icon_ai_2.jpg").write(
                        message["content"]
                    )
            elif isinstance(message["content"], dict):
                # with st.session_state["chat_container"]: #NOUVEAU
                st.chat_message(message["role"], avatar="assets/icon_ai_2.jpg").json(
                    message["content"]
                )


def export_to_notion(text):
    """
    This function when called takes the last message in session state and sends it for it to be converted to a
    notion page with appropriate name
    """
    from langchain_core.prompts import PromptTemplate

    from src.aux_utils.notion_utils import NotionAgent
    from src.main_utils.generation_utils_v2 import LLM_answer_v3

    notion_agent = NotionAgent()

    # load the meeting summarization prompt from prompts/meeting_summary_prompt.txt
    with open("prompts/text2title.txt", "r", encoding="utf-8") as f:
        template = f.read()

    template = PromptTemplate.from_template(template)
    full_prompt = template.format(text=text)
    # give it to the LLM
    with st.spinner("Generating title for Notion page...", show_time=True):
        title = LLM_answer_v3(
            prompt=full_prompt,
            stream=False,
            model_name="llama-3.3-70b-versatile",
            llm_provider="groq",
            temperature=st.session_state["streamlit_config"]["temperature"],
        )

    page_id = notion_agent.create_page_from_markdown(text, page_title=title)
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

    with st.spinner("Transcribing audio...🎤", show_time=True):
        yt = YouTubeTranscriber(chunk_size=st.session_state["config"]["transcription_chunk_size"])
        transcription = yt.transcribe(
            "temp/recorded_audio.wav",
            method="groq",
            diarization=st.session_state["diarization_enabled"],
        )

    print("TRANSCRIPTION: ", transcription)

    st.chat_message("assistant").write("🎤 Audio recorded and transcribed: \n\n" + transcription)
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
        with st.spinner("Searching relevant documents and formulating answer...", show_time=True):
            answer, sources = rag_agent.advanced_RAG_answer(query)
            docs = []

    else:
        from src.main_utils.utils import extract_url

        # we check if the query contains the command @add to add a document to the vectorstore
        if "@add" in query:
            # we create a clean query without the @add command
            if extract_url(query) != None:
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
                st.toast("No url detected, processing the pasted rescource...", icon="🔍")
                from src.main_utils.link_gestion import ExternalKnowledgeManager

                link_manager = ExternalKnowledgeManager(config, client=rag_agent.client)
                link_manager.extract_rescource_from_raw_text(query)  # extract raw text resource
                link_manager.index_rescource()

                pass

        with st.spinner("Determining query intent 🧠 ...", show_time=True):
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
            # Initialize EmailAgent with config and RAG client
            email_agent = EmailAgent(config=config, rag_client=rag_agent.client)
            
            # Use the integrated methods to handle the query
            with st.spinner("Processing email query...", show_time=True):
                answer, docs, sources = email_agent.act(query)


        elif intent == "job search assistance":
                # Initialize JobAgent with current LLM configuration and RAG client
                job_agent = JobAgent(config=config,
                    qdrant_client=rag_agent.client
                )

                # Process the job search query and get RAG answer
                with st.spinner("🔍 Searching and analyzing job listings...",show_time=True):
                    answer, docs, sources = job_agent.act(query)
                    if not answer:
                        st.error("Failed to process job search query")
                        return

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
            from src.aux_utils.meeting_utils import MeetingAgent

            # Initialize MeetingAgent with current LLM configuration and RAG client
            meeting_agent = MeetingAgent(
                config=config,
                qdrant_client=rag_agent.client
            )

            # Process the meeting transcription
            with st.spinner("Generating the meeting summary...", show_time=True):
                summary, file_path, notion_page_id = meeting_agent.process_meeting(
                    transcription=st.session_state["audio_transcription"],
                    user_query=query
                )
                
                if notion_page_id:
                    st.toast("Notion page created!", icon="📋")
                
                st.toast("Meeting summary saved!", icon="🎉")
                st.toast("Meeting summary indexed!", icon="📋")
                
                # Clear the transcription now that we're done with it
                del st.session_state["audio_transcription"]
                
                # Transform the answer into a stream for consistent UI
                answer = "📋 Meeting summary saved in your Notion workspace and in the database!"
                answer = (line for line in answer.split("\n"))

        # elif intent == "sheet or table info extraction":
        #     config["data_sources"] = ["sheets"]
        #     config["enable_source_filter"] = True
        #     config["field_filter"] = ["sheets"]
        #     # put back config to the rag agent
        #     rag_agent.merged_config = config
        #     # initialize the vector agent
        #     from src.main_utils.vectorstore_utils_v4 import VectorAgent

        #     vector_agent = VectorAgent(
        #         default_config=st.session_state["config"],
        #         qdrant_client=rag_agent.client,
        #     )
        #     # remove all the current files of data/sheets
        #     vector_agent.delete(folders=["data/sheets"])
        #     # fetch all sheets using sheet agent
        #     from src.aux_utils.google_sheets_agent import GoogleSheetsAgent

        #     sheet_agent = GoogleSheetsAgent(
        #         credentials_path="google_json_key/python-sheets-446015-aa8eef72c872.json",
        #         save_path="data/sheets",
        #         temp_path="temp",
        #     )
        #     with st.spinner("Fetching the sheets..."):
        #         sheet_agent.fetch_and_save(spreadsheet_name="RechercheEmploi")
        #     # fill the vectorstore with the new sheets
        #     vector_agent.fill()
        #     # answer the query
        #     with st.spinner("Searching relevant documents and formulating answer 📄 ..."):
        #         answer, docs, sources = rag_agent.RAG_answer(query)
    
        elif intent=="table or sheet modification or completion":
            from src.aux_utils.google_sheets_agent import GoogleSheetsAgent
            
            sheet_agent=GoogleSheetsAgent(
            credentials_path='google_json_key/python-sheets-key.json',
            save_path='data/sheets',
            temp_path='temp',
            model_name=config["model_name"], # Updated to use config["model_name"]
            llm_provider=config["llm_provider"]
            #"google" #'groq'
        )
            #act with the agent
            with st.spinner("Modify the sheet...",show_time=True):
                success = sheet_agent.act(query)
                st.toast("Modified the sheet with success: " + str(success))
            
            answer='succes'
            sources=[]
            docs=[]
            #transform this into a generator
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
            with st.spinner("Searching relevant documents and formulating answer 📄 ..."):
                answer, docs, sources = rag_agent.RAG_answer(query)

        #we stream write the answer produced
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

        entry = sources_dict.get(
            absolute_path, {"Filename": filename, "Count": 0, "Emoji": emoji, "Chunks": []}
        )

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
            """,
        ):
            for i, (path, meta) in enumerate(sources_dict.items()):
                cols = st.columns([0.1, 0.5, 0.5, 0.1])
                with cols[0]:
                    st.write(meta["Emoji"])
                with cols[1]:
                    st.write(meta["Filename"])
                with cols[2]:
                    st.write(f"({meta['Count']})")
                with cols[3]:
                    # Utilisez l'ID du premier chunk comme ancre (si disponible)
                    first_chunk_id = (
                        meta["Chunks"][0].metadata.get("_id", "") if meta["Chunks"] else ""
                    )
                    if st.button("🔍", key=f"button_{i}"):
                        switch_page(f"pages/sources_page.py")


if __name__ == "__main__":
    pass
