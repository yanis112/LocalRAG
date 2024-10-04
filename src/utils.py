import json
import logging
import os
import random
import re
import subprocess
import time
from functools import wraps
from typing import AsyncIterator, List

#import easyocr
import numpy as np
import pandas as pd
import spacy
import torch
import yaml
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from keybert import KeyBERT
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_qdrant import Qdrant
from loguru import logger
from lingua import Language, LanguageDetectorBuilder
from moviepy.editor import VideoFileClip
from PIL import Image
from pydub import AudioSegment
from functools import lru_cache

load_dotenv()

os.environ["GROQ_API_KEY"] = (
    "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"
)


def log_execution_time(func):
    """
    Decorator function that logs the execution time of a given function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function.

    Example:
        @log_execution_time
        def my_function():
            # Function code here

        my_function()  # This will log the execution time of my_function
    """
    logger.remove()
    logger.add(
        "execution_time.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
        )
        return result

    return wrapper

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text (str): The text to detect the language of.

    Returns:
        str: 'en' if the text is in English, 'fr' if the text is in French.
    """
    languages = [Language.ENGLISH, Language.FRENCH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    language = detector.detect_language_of(text)
    
    if language == Language.ENGLISH:
        return 'en'
    else:
        return 'fr'

def save_question_answer_pairs(question, answer, sources):
    """Save the question, answer,and sources of a RAG pipeline to a json file.
    Args:
        question (str): The question to save.
        answer (str): The answer to save.
        sources (list): The sources of the answer.
        path (str): The path to save the json file.
    output:
        a json file of structure {"question":question,"answer":answer,"sources":sources [List]}
    """

    PATH = "data_attic/chatbot_history/"
    complete_path = (
        PATH + str(question.replace(" ", "_").replace("?", "")) + ".json"
    )
    with open(complete_path, "w") as f:
        json.dump(
            {"question": question, "answer": answer, "sources": sources}, f
        )


def get_vram_logging():
    """
    return the VRAM used at the moment
    the function is called
    """
    import subprocess

    # Get VRAM usage in GB
    vram_usage = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        .decode()
        .strip()
    )
    vram_usage_gb = (
        float(vram_usage) / 1024
    )  # Convert to GB assuming the output is in MB

    info = f"VRAM usage: {vram_usage_gb:.2f} GB"
    logging.info(info)


def get_k_random_chunks(
    k,
    config,
    get_all=False,
    clone_persist=None,
    clone_embedding_model=None,
    return_embeddings=False,
    special_embedding=None,
):
    """
    Get k random chunks of documents and their corresponding metadata.

    Parameters:
    k (int): The number of random chunks to retrieve.
    config (dict): A dictionary containing configuration settings.
    get_all (bool, optional): If True, return all documents and metadata.
                  If False, return k random documents and metadata.
                  Default is False.
    clone_directory (str, optional): The directory to clone the database from (in case of database cloning).
    clone_embedding_model (str, optional): The embedding model to clone the database from (in case of database cloning).

    Returns:
    tuple: A tuple containing two lists - random_documents and random_metadatas.
        random_documents (list): A list of k random documents.
        random_metadatas (list): A list of k random metadata.

    """
    # we load the embedding model

    if clone_persist is None:
        model_name = config["embedding_model"]
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        if special_embedding is not None:
            encode_kwargs = {
                "normalize_embeddings": True,
                "precision": "float32",
                "batch_size": 1,
            }
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=model_name, encode_kwargs=encode_kwargs
            )

        vectordb = Chroma(
            persist_directory=config["persist_directory"],
            embedding_function=embedding_model,
        )
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=clone_embedding_model
        )
        vectordb = Chroma(
            persist_directory=clone_persist,
            embedding_function=clone_embedding_model,
        )
    # use the get method to get all the documents
    all_documents = vectordb.get()

    # we get a list of metadata
    metadatas = all_documents["metadatas"]
    # print("METADATAS:",metadatas[0:10])
    # we get a list of documents
    documents = all_documents["documents"]

    if get_all:
        return documents, metadatas
    elif return_embeddings:
        return documents, metadatas
    else:
        # we make an identical random sampling on both metadata and documents
        random_indices = random.sample(range(len(documents)), k)
        random_documents = [documents[i] for i in random_indices]
        random_metadatas = [metadatas[i] for i in random_indices]
        return random_documents, random_metadatas


def get_k_random_chunks_qdrant(
    k,
    config,
    get_all=False,
    clone_persist=None,
    clone_embedding_model=None,
    return_embeddings=False,
    special_embedding=None,
    qdrant_db=None,
):
    """
    Get k random chunks of documents and their corresponding metadata.

    Parameters:
    k (int): The number of random chunks to retrieve.
    config (dict): A dictionary containing configuration settings.
    get_all (bool, optional): If True, return all documents and metadata.
                  If False, return k random documents and metadata.
                  Default is False.
    clone_directory (str, optional): The directory to clone the database from (in case of database cloning).
    clone_embedding_model (str, optional): The embedding model to clone the database from (in case of database cloning).

    Returns:
    tuple: A tuple containing two lists - random_documents and random_metadatas.
        random_documents (list): A list of k random documents.
        random_metadatas (list): A list of k random metadata.

    """
    # we load the embedding model

    if clone_persist is None:
        model_name = config["embedding_model"]
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        if special_embedding is not None:
            encode_kwargs = {
                "normalize_embeddings": True,
                "precision": "float32",
                "batch_size": 1,
            }
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=model_name, encode_kwargs=encode_kwargs
            )

        vectordb = qdrant_db

    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=clone_embedding_model
        )
        vectordb = qdrant_db
    # use the similarity_search method to get all the documents
    all_documents = vectordb.similarity_search(query="a", k=1000000)

    # we get a list of metadata
    metadatas = [k.metadata for k in all_documents]
    # print("METADATAS:",metadatas[0:10])
    # we get a list of documents
    documents = [k.page_content for k in all_documents]

    if get_all:
        return documents, metadatas
    elif return_embeddings:
        return documents, metadatas
    else:
        # we make an identical random sampling on both metadata and documents
        random_indices = random.sample(range(len(documents)), k)
        random_documents = [documents[i] for i in random_indices]
        random_metadatas = [metadatas[i] for i in random_indices]
        return random_documents, random_metadatas


def get_all_docs_qdrant(
    raw_database=None,
):
    """
    Get k random chunks of documents and their corresponding metadata.

    Parameters:
    k (int): The number of random chunks to retrieve.
    config (dict): A dictionary containing configuration settings.
    get_all (bool, optional): If True, return all documents and metadata.
                  If False, return k random documents and metadata.
                  Default is False.
    clone_directory (str, optional): The directory to clone the database from (in case of database cloning).
    clone_embedding_model (str, optional): The embedding model to clone the database from (in case of database cloning).

    Returns:
    tuple: A tuple containing two lists - random_documents and random_metadatas.
        random_documents (list): A list of k random documents.
        random_metadatas (list): A list of k random metadata.

    """
    # we load the embedding model

    # use the similarity_search method to get all the documents
    all_documents = raw_database.similarity_search(query="a", k=1000000)
    print("Number of found documents in raw_database:", len(all_documents))
    return all_documents


def zero_shot_classifier(sentence: str, list_classes=["english", "french"]):
    """
    Zero-shot classification of sentences using the Hugging Face pipeline bart-large-mnli.
    input:
        query: the query to classify. type: string
        list_classes: the list of classes to classify the query into. type: list
    output:
        classification: the classification of the query. type: string
    """

    # get the hugging face token from environment variables
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    import requests

    API_URL = (
        "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    )
    headers = {"Authorization": "Bearer " + token}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query(
        {"inputs": sentence, "parameters": {"candidate_labels": list_classes}}
    )

    labels = output["labels"]
    scores = output["scores"]

    # get the index of the maximum score
    index_max_score = scores.index(max(scores))

    # get the corresponding label
    classification = labels[index_max_score]

    return classification



def get_strutured_format(type):
    """
    Return the formatted JSON instance that conforms to the specified type format
    Args:
        type (str): The type of format to generate. Valid values are 'list' and 'dictionary'.
    Returns:
        str: The formatted JSON instance that conforms to the specified JSON schema.
    Raises:
        ValueError: If the specified type is not valid.
   
    """
    
    if type=='list':
        format_instruction = """ The output should be formatted as a JSON instance that conforms to the JSON schema below.

For the given schema:
```json
{
  "properties": {
    "list_steps": {
      "description": "List of strings representing the steps. Example: ['Step 1: Identify the people working on Project A.', 'Step 2: Determine who among them is responsible for maintaining the machines.']",
      "items": {
        "type": "string"
      },
      "title": "List Steps",
      "type": "array"
    }
  },
  "required": ["list_steps"]
}
```

An example of a well-formatted JSON instance that conforms to this schema would be:
```json
{
  "list_steps": [
    "Step 1: Identify the people working on Project A.",
    "Step 2: Determine who among them is responsible for maintaining the machines."
  ]
}
```

An example of a JSON instance that does not conform to this schema would be:
```json
{
    "list_steps": [
        "Step 1: Identify the people working on Project A.",
        "Step 2: Determine who among them is responsible for maintaining the machines."
    ],
    "extra_property": "This is an extra property that is not allowed."
}

Here is the output schema:

 """
 
    if type=='dictionary':
        format_instruction = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:"""

    return format_instruction



@lru_cache(maxsize=None)
def load_translation_model(source, target):
    return GoogleTranslator(source=source, target=target)

@log_execution_time
def translate_to_english(text):
    translator = load_translation_model("french", "english")
    translated_text = translator.translate(text)
    return translated_text

@log_execution_time
def translate_to_french(text):
    translator = load_translation_model("french", "french")
    translated_text = translator.translate(text)
    return translated_text


@log_execution_time
def keywords_extraction(
    doc: str, top_k=1, embedding_model="all-MiniLM-L6-v2"
) -> list:
    """
    Extract keywords from a document
    Input:
        doc: the document to extract keywords from. type: string
    Output:
        keywords: a list of keywords. type: list
        probs: a list of probabilities. type: list
    """

    # translate the document to english
    doc = translate_to_english(doc)

    # delete all characters in the list: [.,!,?,,,;,:]
    doc = re.sub(r"[.,!?\-;:]", "", doc)

    kw_model = KeyBERT(model=embedding_model)
    keywords = kw_model.extract_keywords(doc, top_n=top_k)

    # print("Keywords:",keywords)
    # remove all empty lists inside the keywords list
    keywords = [keyword for keyword in keywords if keyword != []]
    # returns a list of keywords and a list of probabilities
    words = [keyword[0] for keyword in keywords]
    probs = [keyword[1] for keyword in keywords]

    # return only the top_k keywords (if only one keyword is requested, we return the first keyword)
    if top_k == 1:
        return [words[0]], [probs[0]]
    else:
        return words[:top_k], probs[:top_k]


def NER_keyword_extractor(sentence):
    """
    Extract keywords from a sentence using GLiNER model integrated with spaCy.

    Parameters:
    sentence (str): The sentence to extract keywords from.

    Returns:
    list: A list of keywords extracted from the sentence.
    """
    # Configuration for GLiNER integration
    custom_spacy_config = {
        "gliner_model": "urchade/gliner_multi-v2.1",
        "chunk_size": 250,
        "labels": [
            "name" "proper noun",
            "company",
            "acronym",
            "initialism",
            "technical term",
        ],
        "style": "ent",
        "threshold": 0.3,
        "map_location": "cuda"
        if torch.cuda.is_available()
        else "cpu",  # only available in v.0.0.7
    }

    # Initialize a blank English spaCy pipeline and add GLiNER
    nlp = spacy.blank("en")
    if "gliner_spacy" in nlp.pipe_names:
        nlp.remove_pipe("gliner_spacy")
    nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

    # delete all characters in the list: [.,!,?,,,;,:]
    sentence = re.sub(r"[.,!?\-;:]", "", sentence)

    # Process the sentence with the pipeline
    doc = nlp(sentence)

    # Output detected entities
    keywords = [ent.text for ent in doc.ents]

    # if the some keywords are composed of more than one word, we split them
    keywords = [keyword.split() for keyword in keywords]

    # flatten the list to make a list of single words
    keywords = [item for sublist in keywords for item in sublist]

    # remove duplicates
    keywords = list(set(keywords))

    # delete all the word 'who' and 'one' from the query (exceptions)

    if "who" in keywords:
        keywords.remove("who")
    if "one" in keywords:
        keywords.remove("one")

    print("NER KEYWORDS AFTER SPLITTING:", keywords)

    # si keywords est vide, on utilise keybert pour extraire les keywords
    if keywords == []:
        keywords, probs = keywords_extraction(sentence)
        keywords = keywords[0:2]
        print(
            "NER KEYWORDS ARE EMPTY, SWITCHING ON KEYBERT KEYWORDS:", keywords
        )

    return keywords


def token_calculation_prompt(query: str) -> str:
    """
    Return the appropriate token calculation prompt for the query or document.
    """
    coefficient = 1 / 0.45
    num_tokens = len(query.split()) * coefficient
    return num_tokens


def text_preprocessing(text_str):
    """
    Takes the text as input and returns the cleaned text meaning: extra spaces are removed, extra lines are removed, etc...
    Input:
        text_str: the text to clean. type: string
    Output:
        text_str: the cleaned text. type: string
    """

    # remove extra spaces
    text_str = " ".join(text_str.split())
    # remove extra lines (meaning \n\n\n becomes \n same for an arbitrary number of \n)
    text_str = re.sub("\n+", "\n", text_str)
    # remove tabulations and replace them by spaces
    text_str = text_str.replace("\t", " ")

    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002700-\U000027bf"  # Dingbats
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026ff"  # Miscellaneous Symbols
        "\U0001f7e0-\U0001f7ff"  # Geometric Shapes Extended
        "\U0001f980-\U0001f9e0"  # Supplemental Symbols and Pictographs
        "]+",
        flags=re.UNICODE,
    )

    text_str = emoji_pattern.sub(r"", text_str)

    # remove words starting with hashtag
    text_str = re.sub(r"\#\w+", "", text_str)

    # delete all the urls in the text (an url is a sequence of continuous characters starting with http. #http, []htttp can be a url begining !
    text_str = re.sub(r"http\S+", "", text_str)

    # delete 'has joined the channel' occurences (+ name beforeit
    text_str = re.sub(r"\S+ has joined the channel\.", "", text_str)

    return text_str


def structured_excel_loader(path):
    """
    Takes the path of an excel file as input and returns a list of dictionaries where each dictionary corresponds to a row in the excel file.
    Input:
        path: the path of the excel file. type: string
    Output:
        a str representing a json file representing the excel file. type: str
    """
    data = pd.read_excel(path)
    json_file = data.to_json(force_ascii=False)
    # convert the json file to a string
    json_file = "<EXCEL FILE> " + str(json_file) + " </EXCEL FILE>"
    return json_file


class StructuredExcelLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def load(self):  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        json_file = structured_excel_loader(self.file_path)
        doc = Document(
            page_content=json_file,
            metadata={"source": self.file_path},
        )
        return [doc]  # To ensure coherence with the others loaders !!

    # alazy_load is OPTIONAL.
    # If you leave out the implementation, a default implementation which delegates to lazy_load will be used!
    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:  # <-- Does not take any arguments
        """An async lazy loader that reads a file line by line."""
        json_file = structured_excel_loader(self.file_path)
        yield Document(
            page_content=json_file,
            metadata={"source": self.file_path},
        )


class StructuredAudioLoader:
    def __init__(self, file_path: str, chunk_length_ms: int = 30000) -> None:
        self.file_path = file_path
        self.chunk_length_ms = chunk_length_ms

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"

        model_size = "large-v3"
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def extract_audio_from_video(self, file_path: str) -> str:
        video = VideoFileClip(file_path)
        audio_file_path = file_path.rsplit(".", 1)[0] + ".wav"
        video.audio.write_audiofile(audio_file_path)
        return audio_file_path

    def transcribe_audio(self, file_path: str) -> List[str]:
        audio = AudioSegment.from_file(file_path)

        chunks = [
            audio[i : i + self.chunk_length_ms]
            for i in range(0, len(audio), self.chunk_length_ms)
        ]
        print("Audio was split into", len(chunks), "chunks")

        texts = []
        for chunk in chunks:
            # Ensure 'temp' directory exists
            if not os.path.exists("temp"):
                os.makedirs("temp")
            # Overwrite 'temp.wav' each time
            chunk.export("temp/temp.wav", format="wav")
            segments, info = self.model.transcribe("temp/temp.wav", beam_size=5)
            # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            for segment in segments:
                print(
                    "[%.2fs -> %.2fs] %s"
                    % (segment.start, segment.end, segment.text)
                )
                texts.append(segment.text)

        return texts

    @log_execution_time
    def load(self) -> Document:
        file_path = self.file_path
        if self.file_path.endswith((".mp4", ".mpeg")):
            file_path = self.extract_audio_from_video(self.file_path)
        elif self.file_path.endswith((".m4a", ".mp3", ".wav")):
            file_path = (
                self.file_path
            )  # no need to extract audio, it's already an audio file

        texts = self.transcribe_audio(file_path)

        # delete the temporary audio file whose name is temp.wav
        os.remove("temp/temp.wav")

        # remove the model from the GPU by deleting it
        del self.model

        full_text = " ".join(texts)
        doc = Document(
            page_content=full_text, metadata={"source": self.file_path}
        )

        return doc


class StructuredAudioLoaderV2:
    def __init__(
        self,
        file_path: str,
        batch_size=4,
        diarization=False,
        language=None,  # this perform autodection of the language
    ) -> None:
        load_dotenv()
        self.file_path = file_path
        self.hf_token = os.getenv("PYANNOTE_TOKEN")
        self.batch_size = batch_size
        self.diarization = diarization
        self.language = language

    def transcribe_audio(self) -> List[str]:
        # Ensure 'temp' directory exists
        if not os.path.exists("temp"):
            os.makedirs("temp")

        if self.diarization:
            print("DIARIZATION ENABLED")
            # Prepare the command
            command = [
                "insanely-fast-whisper",
                "--model-name",
                "openai/whisper-large-v3",
                "--file-name",
                self.file_path,
                "--flash",
                "True",
                "--hf-token",
                self.hf_token,
                "--diarization_model",
                "pyannote/speaker-diarization-3.1",
                "--transcript-path",
                "temp/temp_transcript.json",
                "--batch-size",
                str(self.batch_size),
                # "--language",
                # self.language,
            ]
        else:
            print("DIARIZATION DISABLED")
            # Prepare the command
            command = [
                "insanely-fast-whisper",
                "--model-name",
                "openai/whisper-large-v3",
                "--file-name",
                self.file_path,
                "--flash",
                "True",
                "--hf-token",
                self.hf_token,
                "--transcript-path",
                "temp/temp_transcript.json",
                "--batch-size",
                str(self.batch_size),
                # "--language",
                # self.language,
            ]

        try:
            # Run the command
            subprocess.run(command, check=True)

            # Load the transcript
            with open(
                "temp/temp_transcript.json", "r", encoding="utf-8"
            ) as file:
                transcript = json.load(file)

        except Exception as e:
            print("ERROR:", e)
            print("Trying again without diarization...")
            # Prepare the command
            command = [
                "insanely-fast-whisper",
                "--model-name",
                "openai/whisper-large-v3",
                "--file-name",
                self.file_path,
                "--flash",
                "True",
                "--hf-token",
                self.hf_token,
                "--transcript-path",
                "temp/temp_transcript.json",
                "--batch-size",
                str(self.batch_size),
                # "--language",
                # self.language,
            ]
            # Run the command
            subprocess.run(command, check=True)

            # Load the transcript
            with open(
                "temp/temp_transcript.json", "r", encoding="utf-8"
            ) as file:
                transcript = json.load(file)

        # Initialize variables
        texts = []
        current_speaker = None
        current_paragraph = ""

        # Iterate over the speakers
        for speaker in transcript["speakers"]:
            # If the speaker has changed, append the current paragraph to texts and start a new one
            if speaker["speaker"] != current_speaker:
                if current_paragraph:
                    texts.append(current_paragraph)
                current_speaker = speaker["speaker"]
                current_paragraph = str(
                    current_speaker + ": " + speaker["text"]
                )
            else:
                # If the speaker is the same, continue the current paragraph
                current_paragraph += " " + speaker["text"]

        # Append the last paragraph
        if current_paragraph:
            texts.append(current_paragraph.strip())

        # Join the paragraphs with double line breaks
        full_text = "\n\n".join(texts)

        print("FULL TEXT:", full_text)

        return full_text

    def load(self) -> dict:
        texts = self.transcribe_audio()

        # delete the temporary transcript file
        # os.remove("temp/temp_transcript.json")

        doc = {"page_content": texts, "metadata": {"source": self.file_path}}

        return doc


def extract_text_from_image(image_path, return_coordinates=False):
    """
    Extracts text from an image using optical character recognition (OCR).

    Args:
        image_path (str or PIL.Image.Image): The path to the image file or a PIL Image object.

    Returns:
        str: The extracted text from the image.
    """
    if isinstance(image_path, Image.Image):
        image_path = np.array(image_path)

    reader = easyocr.Reader(["fr", "en"], gpu=True)
    results = reader.readtext(image_path)

    if return_coordinates:
        texts = [result[1] for result in results]
        coordinates = [result[0] for result in results]
        return texts, coordinates
    else:
        texts = [result[1] for result in results]
        full_text = " /n ".join(texts)

    return full_text


from PIL import Image


class StructuredPDFOcerizer:
    """
    A class for OCR'ing all images in a PDF file and extracting raw text.

    Attributes:
        pdf_path (str): The path to the PDF file.
    """

    def __init__(self) -> None:
        self.pdf_path = None

    def extract_text(self, pdf_path: str):
        """
        Extracts text from a PDF file, including text from images.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: The extracted text from the PDF, including text from images.
        """
        self.pdf_path = pdf_path
        text_from_pdf = ""
        from pdf2image import convert_from_bytes, convert_from_path

        images = convert_from_path(pdf_path)
        # pdf_file = fitz.open(self.pdf_path)  # STEP 2: Open the PDF file with fitz

        # for page_index in range(len(pdf_file)):  # STEP 3: Iterate over PDF pages
        #     try:
        #         page = pdf_file[page_index]
        #         text_from_pdf += page.get_text() or ""  # Extract text directly from the page

        #         image_list = page.get_images(full=True)
        #         if image_list:
        #             print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        #         else:
        #             print(f"[!] No images found on page {page_index}")

        #         for image_index, img in enumerate(image_list, start=1):
        #             xref = img[0]
        #             base_image = pdf_file.extract_image(xref)
        #             image_bytes = base_image["image"]
        #             print("Image bytes:", image_bytes[:10])
        #             image_obj = Image.open(io.BytesIO(image_bytes))
        #             print("TYPE OF IMAGE OBJ:", type(image_obj))
        #             text_from_pdf += extract_text_from_image(image_obj) + " \n "
        #     except Exception as e:
        #         print(f"[!] Error processing page {page_index}: {e}")

        for image in images:
            text_from_pdf += extract_text_from_image(image) + " \n "

        # pdf_file.close()  # Close the PDF file after processing
        return text_from_pdf


class StructuredImageLoader(BaseLoader):
    """
    A class for loading structured data from an image file.

    Args:
        file_path (str): The path to the image file.

    Attributes:
        file_path (str): The path to the image file.

    Methods:
        load: Loads the structured data from the image file.
        alazy_load: Asynchronously loads the structured data from the image file.

    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self):
        """
        Loads the structured data from the image file.

        Returns:
            list: A list of Document objects containing the extracted structured data.

        """
        image = Image.open(self.file_path)
        text = extract_text_from_image(image)
        doc = Document(
            page_content=text,
            metadata={"source": self.file_path},
        )
        return doc

    async def alazy_load(self) -> AsyncIterator[Document]:
        """
        Asynchronously loads the structured data from the image file.

        Yields:
            Document: A Document object containing the extracted structured data.

        """
        image = Image.open(self.file_path)
        text = extract_text_from_image(image)
        yield Document(
            page_content=text,
            metadata={"source": self.file_path},
        )


if __name__ == "__main__":
    # load the config file
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    from src.embedding_model import get_embedding_model

    # load the embedding model
    embedding_model = get_embedding_model(model_name=config["embedding_model"])

    qdrant_db = Qdrant.from_existing_collection(
        path=config["persist_directory"],
        embedding=embedding_model,
        collection_name="qdrant_vectorstore",
    )

    print("QDRANT DB:", qdrant_db)

    start_time = time.time()

    # get k random chunks using Qdrant
    documents, metadatas = get_k_random_chunks_qdrant(
        k=10, config=config, get_all=False, qdrant_db=qdrant_db
    )

    end_time = time.time()

    print("DOCUMENTS:", documents)
    print("METADATAS:", metadatas)

    print("TOTAL Execution Time:", end_time - start_time)
    print("NUMBER OF DOCUMENTS:", len(documents))
