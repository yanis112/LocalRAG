from gradio_client import Client, file
import os
import shutil  # Import the shutil module
from src.aux_utils.transcription_utils import YouTubeTranscriber

class VoiceCloningAgent:
    def __init__(self, endpoint_url="fishaudio/fish-speech-1"):
        """
        Initializes the VoiceCloningAgent with the given endpoint URL.

        Args:
            endpoint_url (str): The URL of the Gradio endpoint for voice cloning.
        """
        self.client = Client(endpoint_url)
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

    def clone_voice(self, text, reference_audio_path, output_dir='cloned_voices', normalize=False, max_new_tokens=1024, chunk_length=200, top_p=0.7, repetition_penalty=1.2, temperature=0.7, seed=0, use_memory_cache="never"):
        """
        Clones a voice based on the reference audio and generates speech for the given text.

        Args:
            text (str): The text to be spoken by the cloned voice.
            reference_audio_path (str): Path to the reference audio file.
            output_dir (str): The directory where the cloned audio will be saved.
            normalize (bool): Whether to use text normalization.
            max_new_tokens (float): Maximum tokens per batch.
            chunk_length (float): Iterative prompt length.
            top_p (float): Top-P sampling parameter.
            repetition_penalty (float): Repetition penalty.
            temperature (float): Temperature for sampling.
            seed (float): Random seed.
            use_memory_cache (str): Memory cache usage.

        Returns:
            tuple: A tuple containing the filepath of the generated audio (in the specified output directory) and any error message.
        """

        # Get the reference text via the transcribe function
        reference_text = self.transcribe_audio(reference_audio_path)

        # Normalize the text if needed
        if normalize:
            text = self.normalize_text(text, use_normalization=normalize)

        # Ensure the reference audio file exists
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")

        # Make API call to clone voice
        result = self.client.predict(
            text=text,
            normalize=normalize,
            reference_audio=file(reference_audio_path),
            reference_text=reference_text,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            use_memory_cache=use_memory_cache,
            api_name="/inference_wrapper"
        )

        cloned_audio_path, error_message = result

        # Move the cloned audio to the specified output directory
        if cloned_audio_path and os.path.exists(cloned_audio_path):
          
            output_filename = os.path.basename(cloned_audio_path)
            destination_path = os.path.join(output_dir, output_filename)
            shutil.move(cloned_audio_path, destination_path)
            cloned_audio_path = destination_path 

        return cloned_audio_path, error_message

# Example usage:
if __name__ == "__main__":
    agent = VoiceCloningAgent()

    # Example 1: Clone voice using a local audio file and save to a specific directory
    try:
        reference_audio = "subprojects/movie_studio/voice_library/woman_narator.mp3"  # Replace with your reference audio file
        text_to_speak = "For generations, they ruled the skies, masters of fire and shadow, Valyria, a name whispered with awe, a power unmatched. Their ambition knew no bounds; they bent dragons to their will, shaped the land itself to their desires, and delved into magics best left undisturbed. But power, unchecked, breeds arrogance, and arrogance invites its own destruction. The earth trembled, the mountains unleashed their fury, and in one cataclysmic night, the greatest empire the world had ever known was reduced to ash. This is not a story of heroes or villains, this is a warning, of what happens when hubris dares to challenge the very nature of the world. The Doom of Valyria, the legends say the dragons were destroyed, but some may have survived... and what do you do when the dragons return?"
        output_directory = "cloned_voices" # Specify your desired output directory

        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        cloned_audio, error_message = agent.clone_voice(text_to_speak, reference_audio)

        if cloned_audio:
            print(f"Cloned audio saved to: {cloned_audio}")
        if error_message:
            print(f"Error: {error_message}")

    except FileNotFoundError as e:
        print(e)