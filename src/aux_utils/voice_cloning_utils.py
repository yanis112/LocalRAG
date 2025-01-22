from gradio_client import Client, file
import os
import shutil  # Import the shutil module
from src.aux_utils.transcription_utils import YouTubeTranscriber
import os 


class VoiceCloningAgent:
    def __init__(self, method="fish_speech"):
        self.method = method
        # Use full Hugging Face space URL for XTTS
        self.endpoint_url = ("fishaudio/fish-speech-1" if method == "fish_speech" 
                           else "https://coqui-xtts.hf.space/--replicas/5891u/")
        self.client = Client(self.endpoint_url)
        self.yt_transcriber = YouTubeTranscriber()


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
    
    def clone_voice(self, text, reference_audio_path, output_dir='cloned_voices', **kwargs):
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")

        os.makedirs(output_dir, exist_ok=True)

        if self.method == "fish_speech":
            cloned_audio_path, error_message = self.clone_voice_fish_speech(text, reference_audio_path, **kwargs)
        else:
            cloned_audio_path, error_message = self.clone_voice_xtts(text, reference_audio_path, **kwargs)

        if cloned_audio_path and os.path.exists(cloned_audio_path):
            output_filename = os.path.basename(cloned_audio_path)
            destination_path = os.path.join(output_dir, output_filename)
            shutil.move(cloned_audio_path, destination_path)
            cloned_audio_path = destination_path

        return cloned_audio_path, error_message


if __name__ == "__main__":
    agent = VoiceCloningAgent(method="xtts_v2")

    try:
        reference_audio = "subprojects/movie_studio/voice_library/agamemnon.mp3"
        text_to_speak = "Burn Guiss ! Burn it for the glory of Valyria !"
        output_directory = "cloned_voices"

        cloned_audio, error_message = agent.clone_voice(
            text_to_speak, 
            reference_audio,
            output_dir=output_directory,
            language="en,en",
            use_mic=False,
            cleanup_voice=True,
            no_lang_auto_detect=True
        )

        if cloned_audio:
            print(f"Cloned audio saved to: {cloned_audio}")
        if error_message:
            print(f"Error: {error_message}")

    except FileNotFoundError as e:
        print(e)