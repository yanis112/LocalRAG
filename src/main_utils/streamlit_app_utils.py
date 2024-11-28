import os
import textwrap
import time
from pathlib import Path

from litellm import transcription
import streamlit as st
import yaml

from src.main_utils.generation_utils_v2 import LLM_answer_v3


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


@st.fragment
def handle_uploaded_file(uploaded_file):
    """
    Handles the uploaded file and processes it based on its extension.
    Parameters:
    uploaded_file (UploadedFile): The file uploaded by the user.
    Returns:
    bytes: The data of the processed file if applicable.
    Processes:
    - For audio files (mp3, wav, m4a, mp4):
        - Saves the audio file as a wav file.
        - Transcribes the audio using YouTubeTranscriber.
        - Creates a transcription text file.
        - Displays success messages and returns the transcription data.
    - For image files (png, jpg, jpeg):
        - Displays the image.
        - Saves the image in the temp folder.
        - Analyzes the image using UniversalImageLoader.
        - Displays success messages and the extracted content from the image.
    - For PDF files (pdf):
        - Saves the uploaded PDF.
        - Performs OCR on the PDF using StructuredPDFOcerizer.
        - Displays success messages and the extracted text from the PDF.
    """

    temp_dir = "temp"
    extension = str(Path(uploaded_file.name).suffix)

    if extension in [".mp3", ".wav", ".m4a", ".mp4"]:
        from src.aux_utils.transcription_utils import YouTubeTranscriber

        file_path = save_audio_as_wav(uploaded_file, temp_dir)
        #st.success(f"File saved as {file_path}")

        with st.spinner("Transcribing audio 🎤 ..."):
            yt = YouTubeTranscriber()
            transcription = yt.transcribe(file_path, method="groq")

        #print("TRANSCRIPTION: ", transcription)
        st.toast("Transcription successful !", icon="🎤")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Here is your transcribed audio 🔉📜 ! \n\n"
                + str(transcription),
            }
        )

      
        st.session_state["uploaded_file"] = True
        st.session_state["audio_transcription"] = transcription
        
        # txt = create_transcription_txt(transcription)
        # print("TXT FILE CREATED !")

        # with open(txt, "rb") as f:
        #     data = f.read()

        # st.toast("Transcription successful!", icon="🎉")
        # print("DOWNLOAD BUTTON CREATED !")

        # return data

    elif extension in [".png", ".jpg", ".jpeg"]:
        st.image(uploaded_file)
        # save the image in the temp folder
        uploaded_file.seek(0)
        with open("temp/" + uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        from src.aux_utils.image_analysis import UniversalImageLoader

        with st.spinner(
            "Analyzing the image... it should take less than 2 minutes 😜"
        ):
            # load the universal image loader
            structured_output = UniversalImageLoader().universal_extract(
                image_path="temp/" + uploaded_file.name
            )
            print("TYPE OF STRUCTURED OUTPUT:", type(structured_output))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Here is the content i extracted from your image 🖼️: \n\n"
                    + str(structured_output),
                }
            )
            st.session_state["uploaded_file"] = True

        st.toast("Image analysis successfull !", icon="🎉")

    elif extension in [".pdf"]:
        from src.main_utils.utils import StructuredPDFOcerizer

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
    from streamlit_pills import stp

    selected = stp.pills(
        label="Suggestions", options=list_suggestions, index=None
    )
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
    from fpdf import FPDF

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


def display_chat_history():
    for message in st.session_state.messages:
        if isinstance(message["content"], str):
            if (
                message["role"] == "assistant"
                and "<output>" in message["content"]
            ):
                # Split the content at the <output> tag
                reasoning, final_output = message["content"].split(
                    "<output>", 1
                )

                # Display the final output
                with st.chat_message(
                    message["role"], avatar="assets/icon_ai_2.jpg"
                ):
                    st.write(final_output.strip())

                    # Add an expander for the reasoning
                    with st.expander("Show intermediate reasoning steps"):
                        st.write(reasoning.strip())
            else:
                # For all other messages, display normally
                st.chat_message(
                    message["role"], avatar="assets/icon_ai_2.jpg"
                ).write(message["content"])
        elif isinstance(message["content"], dict):
            st.chat_message(
                message["role"], avatar="assets/icon_ai_2.jpg"
            ).json(message["content"])


def clear_chat_history():
    st.session_state.messages = []
    st.toast("Chat history cleared!", icon="🧹")
    #rerun the app to clear the chat
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
    from src.aux_utils.transcription_utils import YouTubeTranscriber

    if not os.path.exists("temp"):
        os.makedirs("temp")

    with open("temp/recorded_audio.wav", "wb") as f:
        f.write(audio.getbuffer())

    print("Audio saved as recorded_audio.wav")

    with st.spinner("Transcribing audio...🎤"):
        yt = YouTubeTranscriber()
        transcription = yt.transcribe("temp/recorded_audio.wav", method="groq")

    print("TRANSCRIPTION: ", transcription)

    st.chat_message("assistant").write(
        "🎤 Audio recorded and transcribed: \n\n" + transcription
    )
    st.toast("Transcription successful!", icon="🎉")
    
    #add transcription to the session state
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

    # we load the intent classifier into a session state variable so that we check if the intent is already loaded

    if "intent_classifier" not in st.session_state:
        st.session_state["intent_classifier"] = IntentClassifier(labels_dict=config["actions_dict"])

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
        with st.spinner(
            "Searching relevant documents and formulating answer..."
        ):
            answer, sources = rag_agent.advanced_RAG_answer(query)
            docs = []

    else:
        
        #we check if the query contains the command @add to add a document to the vectorstore
        if "@add" in query:
            #we create a clean query without the @add command
            query=query.replace("@add","")
            #we import the external knowledge manager
            from src.main_utils.link_gestion import ExternalKnowledgeManager
            link_manager=ExternalKnowledgeManager(config,client=rag_agent.client)
            link_manager.extract_rescource(query)
            link_manager.index_rescource()
            #we stop the process here
            st.toast("New rescource indexed !", icon="🎉")
            return None
        
        with st.spinner("Determining query intent 🧠 ..."):
            # we detect the intent of the query
            intent = st.session_state["intent_classifier"].classify(query,method="LLM")
            print("Intent detected: ", intent)
            st.toast("Intent detected: " + intent, icon="🧠")

        if intent == "employer contact writing":
            from src.aux_utils.auto_job import auto_job_writter

            with st.spinner("Generating a text for a job offer..."):
                answer = LLM_answer_v3(
                    prompt=auto_job_writter(query, "info.yaml", "cv.txt"),
                    stream=True,
                    model_name=config["model_name"],
                    llm_provider=config["llm_provider"],
                )
                sources = []
        elif intent == "email support inquiry":
            # fetch the last 100 emails and search for the answer
            from src.aux_utils.email_utils import EmailAgent

            with st.spinner("Fetching new emails..."):
                email_utils = EmailAgent()
                #email_utils.connect()
                email_utils.fetch_new_emails(last_k=100)
                #email_utils.disconnect()

            # fill the vectorstore withg the new emails
            from src.main_utils.vectorstore_utils_v2 import VectorAgent

            with st.spinner("Filling the vectorstore with new emails..."):
                agent = VectorAgent(
                    default_config=config, qdrant_client=rag_agent.client
                )
                agent.fill()
                print("Vectorstore filled with new emails !")

            config["data_sources"] = ["emails"]
            config["enable_source_filter"] = True
            config["field_filter"] = ["emails"]
            with st.spinner(
                "Searching relevant documents and formulating answer 📄 ..."
            ):
                answer, docs, sources = rag_agent.RAG_answer(query)

        elif intent == "job search assistance":
            from src.aux_utils.job_scrapper import JobAgent

            config["data_sources"] = ["jobs"]
            config["enable_source_filter"] = True
            config["field_filter"] = ["jobs"]

            # scrapping jobs
            with st.spinner("Scraping job offers..."):
                try:
                    job_scrapper = JobAgent(
                        search_term="Data Scientist",
                        location="Aix en Provence",
                        hours_old=200,
                        results_wanted=20,
                        google_search_term="data scientist aix en provence",
                        is_remote=False,
                    )
                    job_scrapper.scrape_and_convert()
                except Exception as e:
                    print("EXCEPTION IN JOB SCRAPPING:", e)
                    st.error(
                        "An error occured while scrapping the job offers !"
                    )

            # fill the vectorstore with the new job offers
            from src.main_utils.vectorstore_utils_v2 import VectorAgent

            with st.spinner("Filling the vectorstore with new job offers..."):
                agent = VectorAgent(
                    default_config=config, qdrant_client=rag_agent.client
                )
                agent.fill()
                print("Vectorstore filled with new job offers !")

            with st.spinner(
                "Searching relevant jobs and formulating answer 📄 ..."
            ):
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
                    model_name="Meta-Llama-3.1-405B-Instruct",
                    llm_provider="github",
                )
                output_path = graph_maker.generate_graph(base_prompt=query)
                # show the svg file in the streamlit app
                st.image(output_path)
                sources = []
                answer = "Here is the generated graph !"
                # convert answer to a generator like object to be treated as a stream
                answer = (line for line in [answer])

        elif intent == "meeting notes summarization" :
            from langchain_core.prompts import PromptTemplate
            # load the meeting summarization prompt from prompts/meeting_summary_prompt.txt
            with open("prompts/meeting_summary_prompt.txt", "r", encoding='utf-8') as f:
                template = f.read()
                
            template= PromptTemplate.from_template(template)
            full_prompt=template.format(transcription=st.session_state["audio_transcription"],user_query=query)
               
            # give it to the LLM
            with st.spinner("Generating the meeting summary..."):
                raw_answer = LLM_answer_v3(
                    prompt=full_prompt,
                    stream=False,
                    model_name=config["model_name"],
                    llm_provider=config["llm_provider"],
                )
                sources = []
                #transform the answer into a stream / generator object
                answer = (line for line in raw_answer.split("\n"))
                #save the transcription in a file, the name of the file is the date of the meeting (today)
                
                with open("data/meeting_summary/"+str(time.strftime("%Y-%m-%d"))+".txt", "w", encoding='utf-8') as f:
                    #write a header with the date at the beginning of the file
                    f.write("Meeting summary for the date: "+str(time.strftime("%Y-%m-%d"))+"\n\n")
                    f.write(raw_answer)
                    
                #add a toast to notify the user that the meeting summary has been saved
                st.toast("Meeting summary saved !", icon="🎉")
                #fill the vectorstore with the new meeting summary
                from src.main_utils.vectorstore_utils_v2 import VectorAgent
                vector_agent = VectorAgent(default_config=st.session_state["config"],qdrant_client=rag_agent.client)
                vector_agent.fill()
                print("Vectorstore filled with new meeting summary !")
                st.toast("Meeting summary indexed !", icon="📋")
                #put st.session_state["audio_transcription"] to None now that the meeting summary has been saved
                del st.session_state["audio_transcription"]
                    
                

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
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Add the sources list to a session state variable
        st.session_state["sources"] = sources

        display_sources(sources)


@st.fragment
def display_sources(sources):
    import os
    import webbrowser

    from streamlit_extras.stylable_container import stylable_container

    source_counts = {source: sources.count(source) for source in sources}

    # Emoji dictionary (you can customize this)
    subdomain_emojis = {
        "jobs": "💼",
        "politique": "🏛️",
        "internet": "🌐",
        "emails": "📧",
    }

    data = []
    for source in sources:
        absolute_path = os.path.abspath(source)
        filename = os.path.basename(absolute_path)

        # Extract the subdomain from the path
        subdomain = None
        for domain in subdomain_emojis.keys():
            if domain in absolute_path.lower():
                subdomain = domain
                break
        emoji = subdomain_emojis.get(
            subdomain, "📄"
        )  # Default emoji if subdomain not found

        if absolute_path not in [d["Absolute Path"] for d in data]:
            data.append(
                {
                    "Absolute Path": absolute_path,
                    "Filename": filename,
                    "Count": source_counts[source],
                    "Emoji": emoji,
                }
            )

    with st.expander("Sources 📑"):
        with stylable_container(
            key="sources_container",
            css_styles="""
            #sources_container .stButton { /* Targets buttons ONLY within the container */
                background-color: transparent;
                border: none;
                padding: 0;
            }
            #sources_container .stButton:hover {
                background-color: transparent;
            }
            #sources_container .stButton > svg {
                fill: #99A3BA;
            }
            #sources_container .stButton:hover > svg {
                fill: #31333F; 
            }
            #sources_container .element-container { /* Targets source containers */
                background-color: #f5f5f5;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 5px;
            }
            """,
        ):
            for i, item in enumerate(data):
                col1, col2, col3, col4 = st.columns(
                    [0.1, 0.5, 0.5, 0.1]
                )  # Added emoji column back

                with col1:
                    st.write(item["Emoji"])  # Display the emoji

                with col2:
                    st.write(item["Filename"])

                with col3:
                    st.write(f"({item['Count']})")

                with col4:
                    if st.button("🔍", key=f"button_{i}"):
                        webbrowser.open(
                            f"file://{item['Absolute Path']}", new=1
                        )


if __name__ == "__main__":
    pass
