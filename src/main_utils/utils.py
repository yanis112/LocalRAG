import json
import os
import re
import subprocess
from functools import lru_cache
from typing import List

# import easyocr
from dotenv import load_dotenv
from src.aux_utils.logging_utils import log_execution_time

load_dotenv()


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text (str): The text to detect the language of.

    Returns:
        str: 'en' if the text is in English, 'fr' if the text is in French.

    """
    # lazy loading of the module
    from lingua import Language, LanguageDetectorBuilder

    languages = [Language.ENGLISH, Language.FRENCH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    language = detector.detect_language_of(text)

    if language == Language.ENGLISH:
        return "en"
    else:
        return "fr"


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

    if type == "list":
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

    if type == "dictionary":
        format_instruction = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:"""

    return format_instruction


@lru_cache(maxsize=None)
def load_translation_model(source, target):
    from deep_translator import GoogleTranslator

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


if __name__ == "__main__":
    pass
