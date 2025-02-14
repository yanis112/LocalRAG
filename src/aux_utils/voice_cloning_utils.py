import logging
import os
import shutil
import base64
import time
from functools import wraps
from src.aux_utils.transcription_utils import YouTubeTranscriber
from zyphra import ZyphraClient, ZyphraError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def retry_on_timeout(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except ZyphraError as e:
                    if e.status_code == 524:  # Timeout error
                        retries += 1
                        if retries < max_retries:
                            delay = base_delay * (2 ** (retries - 1))  # Exponential backoff
                            logger.warning(f"Timeout occurred. Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
                            time.sleep(delay)
                            continue
                    raise
            return func(*args, **kwargs)
        return wrapper
    return decorator

class VoiceCloneError(Exception):
    """Custom exception for voice cloning errors"""
    pass

class VoiceCloningAgent:
    def __init__(self, method="zonos"):
        self.method = method
        self.yt_transcriber = YouTubeTranscriber()
        
        if self.method == "zonos":
            try:
                # Initialize Zonos client with API key from environment variable
                self.client = ZyphraClient(api_key=os.getenv("ZONOS_API_KEY"))
                logger.info("Successfully initialized Zonos API client")
            except Exception as e:
                logger.error(f"Failed to initialize Zonos client: {str(e)}")
                raise VoiceCloneError(f"Failed to initialize voice cloning client: {str(e)}")
        else:
            # Initialize Hugging Face client for other methods
            try:
                from gradio_client import Client
                self.endpoint_url = {
                    "fish_speech": "fishaudio/fish-speech-1",
                    "xtts": "https://coqui-xtts.hf.space/--replicas/5891u/"
                }.get(method, "fishaudio/fish-speech-1")
                self.client = Client(self.endpoint_url, hf_token=os.getenv("HUGGINGFACE_TOKEN"))
                logger.info(f"Successfully connected to endpoint: {self.endpoint_url}")
            except Exception as e:
                logger.error(f"Failed to initialize client for {self.endpoint_url}: {str(e)}")
                raise VoiceCloneError(f"Failed to initialize voice cloning client: {str(e)}")

    def normalize_text(self, user_input, use_normalization=False):
        """
        Normalizes the input text using the /normalize_text API.

        Args:
            user_input (str): The text to normalize.
            use_normalization (bool): Whether to use text normalization.

        Returns:
            str: The normalized text.
        """
        result = self.client.predict(
            user_input=user_input,
            use_normalization=use_normalization,
            api_name="/normalize_text"
        )
        return result

    def transcribe_audio(self, audio_filepath):
        """
        Transcribes the audio file using the YouTubeTranscriber.

        Args:
            audio_filepath (str): Path to the audio file.

        Returns:
            str: The transcribed text from the audio.
        """

        transcription = self.yt_transcriber.transcribe(audio_filepath, method="groq")
        return transcription

    def select_example_audio(self, audio_file=""):
        """
        Selects an example audio using the /select_example_audio API.

        Args:
            audio_file (str): The name of the example audio file.

        Returns:
            tuple: A tuple containing the filepath of the selected audio and its transcription.
        """
        result = self.client.predict(
            audio_file=audio_file,
            api_name="/select_example_audio"
        )
        return result

    def clone_voice_fish_speech(self, text, reference_audio_path, **kwargs):
        """Fish Speech implementation of voice cloning"""
        reference_text = self.transcribe_audio(reference_audio_path)
        
        result = self.client.predict(
            text=text,
            normalize=kwargs.get('normalize', False),
            reference_audio=file(reference_audio_path),
            reference_text=reference_text,
            max_new_tokens=kwargs.get('max_new_tokens', 1024),
            chunk_length=kwargs.get('chunk_length', 200),
            top_p=kwargs.get('top_p', 0.7),
            repetition_penalty=kwargs.get('repetition_penalty', 1.2),
            temperature=kwargs.get('temperature', 0.7),
            seed=kwargs.get('seed', 0),
            use_memory_cache=kwargs.get('use_memory_cache', "never"),
            api_name="/inference_wrapper"
        )
        return result
    
    def clone_voice_xtts(self, text, reference_audio_path, **kwargs):
        """XTTS v2 implementation of voice cloning"""
        try:
            result = self.client.predict(
                text,  # Text Prompt
                kwargs.get('language', 'en,en'),  # Language
                reference_audio_path,  # Reference Audio
                reference_audio_path,  # Use Microphone for Reference
                kwargs.get('use_mic', False),  # Use Microphone
                kwargs.get('cleanup_voice', True),  # Cleanup Reference Voice
                kwargs.get('no_lang_auto_detect', True),  # Do not use language auto-detect
                True,  # Agree checkbox
                fn_index=1
            )
            
            if isinstance(result, tuple) and len(result) >= 2:
                return result[1], None  # Return synthesized audio path
            return None, "Invalid API response format"
            
        except Exception as e:
            return None, f"XTTS API error: {str(e)}"
    
    @retry_on_timeout(max_retries=3, base_delay=1)
    def clone_voice_zonos(self, text, reference_audio_path, **kwargs):
        """Voice cloning using direct Zonos API with retry logic"""
        try:
            logger.info(f"Attempting voice cloning with Zonos API")
            logger.info(f"Input text: {text[:50]}...")
            logger.info(f"Reference audio: {reference_audio_path}")
            
            # Split long text into chunks if needed (to prevent timeouts)
            max_chars_per_request = 500
            if len(text) > max_chars_per_request:
                logger.info(f"Text length ({len(text)} chars) exceeds recommended maximum. Splitting into chunks.")
                text_chunks = [text[i:i+max_chars_per_request] for i in range(0, len(text), max_chars_per_request)]
            else:
                text_chunks = [text]
            
            # Read and encode reference audio file
            with open(reference_audio_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            final_audio_data = None
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
                
                # Generate speech with cloned voice
                chunk_audio = self.client.audio.speech.create(
                    text=chunk,
                    speaker_audio=audio_base64,
                    speaking_rate=kwargs.get('speaking_rate', 15),
                    language_iso_code=kwargs.get('language', 'en-us'),
                    mime_type=kwargs.get('mime_type', 'audio/mp3')
                )
                
                if isinstance(chunk_audio, bytes):
                    if final_audio_data is None:
                        final_audio_data = chunk_audio
                    else:
                        # TODO: Implement proper audio concatenation if needed
                        final_audio_data = chunk_audio
            
            # Save the final audio data to a file
            output_file = os.path.join(kwargs.get('output_dir', 'cloned_voices'), 'output.mp3')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'wb') as f:
                f.write(final_audio_data)
            
            logger.info(f"Successfully generated voice clone: {output_file}")
            return output_file, None
                
        except ZyphraError as e:
            error_msg = f"Zonos API error: {e.status_code} - {e.response_text}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def clone_voice(self, text, reference_audio_path, output_dir='cloned_voices', **kwargs):
        try:
            if not os.path.exists(reference_audio_path):
                error_msg = f"Reference audio file not found: {reference_audio_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Starting voice cloning with method: {self.method}")
            if self.method == "zonos":
                cloned_audio_path, error_message = self.clone_voice_zonos(text, reference_audio_path, output_dir=output_dir, **kwargs)
            elif self.method == "fish_speech":
                cloned_audio_path, error_message = self.clone_voice_fish_speech(text, reference_audio_path, **kwargs)
            elif self.method == "xtts":
                cloned_audio_path, error_message = self.clone_voice_xtts(text, reference_audio_path, **kwargs)
            else:
                error_msg = f"Unsupported method: {self.method}"
                logger.error(error_msg)
                return None, error_msg

            if error_message:
                logger.error(f"Error during voice cloning: {error_message}")
                return None, error_message

            if cloned_audio_path and os.path.exists(cloned_audio_path):
                return cloned_audio_path, None
            
            error_msg = "No output file generated"
            logger.error(error_msg)
            return None, error_msg

        except Exception as e:
            error_msg = f"Unexpected error during voice cloning: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg


if __name__ == "__main__":
    try:
        # Example usage with Zonos
        agent = VoiceCloningAgent(method="zonos")
        reference_audio = "subprojects/movie_studio/voice_library/agamemnon.mp3"
        text_to_speak = "Burn Guiss ! Burn it for the glory of Valyria !"
        output_directory = "cloned_voices"

        logger.info("Starting voice cloning test")
        logger.info(f"Reference audio: {reference_audio}")
        logger.info(f"Text to speak: {text_to_speak}")

        cloned_audio, error_message = agent.clone_voice(
            text_to_speak, 
            reference_audio,
            output_dir=output_directory,
            language="en-us",
            model_choice="Zyphra/Zonos-v0.1-hybrid"
        )

        if cloned_audio:
            logger.info(f"Success! Cloned audio saved to: {cloned_audio}")
        if error_message:
            logger.error(f"Failed to clone voice: {error_message}")

    except Exception as e:
        logger.error("Critical error in main execution:", exc_info=True)