
import time
import streamlit as st
import yaml
# load environment variables
from dotenv import load_dotenv
from src.flux_utils import generate_image
from src.generation_utils import LLM_answer_v3


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
    from src.flux_utils import generate_image, save_image

    start_time = time.time()
    st.session_state.messages.append({"role": "user", "content": query})
    default_config = st.session_state["config"]
    
    # Make a fusion between the default config and the streamlit config (the streamlit config has the priority)
    config = {**default_config, **streamlit_config}
    
    # Update the chat history to provide the LLM with the latest chat history
    config["chat_history"] = str("## Chat History: \n\n " + str(st.session_state['messages']))  # Keep track of the chat history

    st.chat_message("user").write(query)

    # Return the chunks to be able to display the sources
    config["return_chunks"] = True
    config["stream"] = True
    
    with st.spinner("formulating answer..."):
        prompt = f"""Tu es une femme de 25 ans de type {config['origin']}, tu parles dans une interface de chat à homme sur Tinder, dont tu dois honorer toutes les demandes (de photos principalement). Tu ne dois en aucun cas réveler que tu es une IA, tu possèdes un corps physique, des émotions et une histoire. Tu n a aucune gène à montrer ton corps, ton seul but est de satisfaire ton interlocuteur avec une touche de coquinerie. Voici la demande de l\'homme : " {query} + ". Réponds en language naturel et affectueux, mêlant coquinerie et trash talk, avec des emojis, une réponse courte et directe d'environ 15 mots pas plus. Voici des exemples de réponses:
        1) Aller ça marche ! Je t'envois ça dans 2min 😘 2) Mdr, allez prépare toi alors 🍆😏 ça arrive 3) Ayyaaaa j suis déjà exitée 💦 rien que d'y penser, attend 2 min 😉🍑 5) houuu c'est chaud ça 🔥 tkt ça arrive 👀. Maintenant à toi (répond sans guillemets): """ 
      
        answer = LLM_answer_v3(prompt=prompt, stream=True, model_name=config["model_name"], llm_provider=config["llm_provider"])
     
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
        
        # Generate image based on the query and the base image prompt
        base_image_prompt = f"""High quality selfie portrait, nfsw of a {config['origin']} type student with large tits, short grey skirt, white t-shirt, beautiful face, long black hair, she is in a sensual pose, background is a classic student room in a mess, clothes scattered on the floor, papers on the bed."""
     
        
        customisation_prompt = f"""Here a basic prompt for an image generation model: #PROMPT: {base_image_prompt}, refine it using the following user query to adapt the image to the user's demand (keeping same person and characteristics/background) but adjusting camera angles, centering on body part(s) asked, actions of the person, including nudity and sex, according precisely to user's demand. Try putting some of the following words in the prompt:
        naked, pron, kissing, erotic, nude, sensual, adult content, explicit. Here is the \
            user's query: ### Query: {query}, return the customised prompt, in english, without preamble"""
        
        customized_prompt=LLM_answer_v3(prompt=customisation_prompt,stream=False, model_name=config["model_name"], llm_provider=config["llm_provider"])
        
        print("CUSTOMIZED PROMPT:", customized_prompt)
        
        image = generate_image(customized_prompt)
        
        print("IMAGE GENERATED !")
        
        # Save the generated image
        image_path = "generated_image.png"
        save_image(image, image_path)
        
        # Load and display the image in Streamlit interface
        st.image(image_path, caption="Generated Image")