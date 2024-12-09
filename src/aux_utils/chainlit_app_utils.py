import time
import streamlit as st

import yaml

from src.main_utils.generation_utils import RAG_answer, advanced_RAG_answer, LLM_answer_v3

async def load_config():
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

    except Exception as e:
        print("EXCEPTION IN LOAD CONFIG:", e)
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
    return config

def process_query(query, config):
    from src.text_classification_utils import IntentClassifier
    start_time = time.time()
    default_config = load_config()
    config = {**default_config, **config}
    classifier = IntentClassifier(config["actions"])
    config["chat_history"] = str("## Chat History: \n\n " + str(st.session_state['messages']))
    if config["deep_search"]:
        pass
    if config["field_filter"] != []:
        config["enable_source_filter"] = True
    else:
        config["enable_source_filter"] = False
    if config["emails_answer"]:
        config["data_sources"] = ["email"]
        config["enable_source_filter"] = True
        config["field_filter"] = ["email"]
    config["return_chunks"] = True
    config["stream"] = True
    if config["deep_search"]:
        answer, sources = advanced_RAG_answer(
            query,
            default_config=default_config,
            config=config,
        )
        docs = []
    else:
        intent = classifier.classify(query)
        if intent == "rediger un texte pour une offre":
            from aux_utils.auto_job import auto_job_writter
            answer = LLM_answer_v3(prompt=auto_job_writter(query, "info.yaml", "cv.txt"), stream=False, model_name=config["model_name"], llm_provider=config["llm_provider"])
            sources = []
        elif intent == "question sur des emails":
            config["data_sources"] = ["emails"]
            config["enable_source_filter"] = True
            config["field_filter"] = ["emails"]
            answer, docs, sources = RAG_answer(
                query,
                default_config=default_config,
                config=config,
            )
        elif intent == "rechercher des offres d'emploi":
            config["data_sources"] = ["jobs"]
            config["enable_source_filter"] = True
            config["field_filter"] = ["jobs"]
            answer, docs, sources = RAG_answer(
                query,
                default_config=default_config,
                config=config,
            )
        elif intent == "write instagram description":
            from aux_utils.auto_instagram_publi import instagram_descr_prompt
            answer = LLM_answer_v3(prompt=instagram_descr_prompt(query), stream=False, model_name=config["model_name"], llm_provider=config["llm_provider"])
            sources = []
        else:
            answer, docs, sources = RAG_answer(
                query,
                default_config=default_config,
                config=config,
            )
        end_time = time.time()
    return answer
