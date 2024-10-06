from gradio_client import Client, file
import shutil
from src.transcription_utils import YouTubeTranscriber

def run_inference(text, reference_audio_path, output_file='output_audio.wav'):
    """
    Run inference using the Gradio client and save the resulting audio locally.

    Parameters:
    text (str): The input text for the inference.
    reference_audio_path (str): The path to the local reference audio file.
    output_file (str): The name of the output audio file. Default is 'output_audio.wav'.
    """
    # Initialize YouTubeTranscriber
    transcriber = YouTubeTranscriber(chunk_size=60, batch_size=1,language='fr')
    
    # Transcribe the reference audio to get the reference text
    reference_text = transcriber.transcribe(reference_audio_path, method='groq')
    reference_text = " ".join(reference_text)  # Join the list of transcriptions into a single string
    
    print(f"Reference text: {reference_text}")
    
    client = Client("fishaudio/fish-speech-1")
    result = client.predict(
        text=text,
        enable_reference_audio=True,
        reference_audio=file(reference_audio_path),
        reference_text=reference_text,
        max_new_tokens=0,
        chunk_length=200,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        batch_infer_num=1,
        if_load_asr_model=False,
        api_name="/inference_wrapper"
    )
    
    # The result is a tuple of 9 elements, we are interested in the first audio file path
    audio_info = result[1]
    
    # Extract the local file path from the dictionary
    if isinstance(audio_info, dict) and 'value' in audio_info:
        local_audio_path = audio_info['value']
        
        # Copy the file to the desired output location
        shutil.copy(local_audio_path, output_file)
        print(f"Audio file saved as {output_file}")
    else:
        print("Failed to retrieve the audio file path from the result.")

# Example usage
if __name__ == "__main__":    
    run_inference(
        text="Bonjour je m'appelle Yanis et j'ai un énorme bangala vaineux.",
        reference_audio_path='voice_ref.wav',
        output_file='output_audio.wav'
    )