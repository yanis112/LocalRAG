import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

def run_diarization(audio_path, auth_token):
    try:
        # Initialize pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )

        # Setup GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        # Load and validate audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Run diarization with progress hook
        with ProgressHook() as hook:
            diarization = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
                min_speakers=1,
                max_speakers=5
            )

        # Write RTTM output
        output_file = audio_path.rsplit('.', 1)[0] + '.rttm'
        with open(output_file, "w") as rttm:
            diarization.write_rttm(rttm)

        print(f"Diarization completed. Output saved to {output_file}")
        return diarization

    except Exception as e:
        print(f"Error during diarization: {str(e)}")
        return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    audio_path = "aux_data/test_audio_diarization.mp3"
    auth_token = os.getenv('HUGGINGFACE_TOKEN')
    
    diar=run_diarization(audio_path, auth_token)
    print(diar)