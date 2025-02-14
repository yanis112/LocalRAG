import os
import shutil
import torch
import torchaudio
import yt_dlp
from pydub import AudioSegment
from groq import Groq
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List
from functools import lru_cache
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

load_dotenv()

""" Various tools and classes to transcribe audios from YouTube videos with diarization support """

class YouTubeTranscriber:
    """
    A class to download and transcribe audio from YouTube videos.

    Supports different transcription methods including Whisper, Groq, and Insanely Fast Whisper.
    Also includes speaker diarization using pyannote.audio if requested.
    """
    def __init__(self, chunk_size: int = 0, batch_size: int = 1, language: str = 'en'):
        """
        Initializes the YouTubeTranscriber with specified parameters.

        Args:
            chunk_size (int): Size of audio chunks in seconds. Default is 0 (no chunking).
            batch_size (int): Batch size for processing audio chunks. Default is 1.
            language (str): Language of the audio. Default is 'en' (English).
        """
        self.chunk_size = chunk_size * 1000  # Convert to milliseconds
        self.batch_size = batch_size
        self.language = language
        self.temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"Temporary directory checked or created: {self.temp_dir}")

    def download_audio(self, url: str, output_dir: str) -> str:
        """
        Downloads audio from a given URL and saves it as an MP3 file.

        Args:
            url (str): The URL of the audio to download.
            output_dir (str): The directory where the downloaded audio file will be saved.

        Returns:
            str: The file path of the downloaded audio file, or None if the download fails.
        """
        print(f"Downloading audio from URL: {url}")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                # info_dict: dictionary containing metadata about the video
                filename = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
                # filename: string representing the path to the downloaded audio file
            print(f"Audio downloaded: {filename}")
            return filename
        except yt_dlp.utils.DownloadError as e:
            print(f"Error downloading audio: {e}")
            return None

    def chunk_audio(self, file_path: str) -> List[str]:
        """
        Splits an audio file into smaller chunks.

        Args:
            file_path (str): The path to the audio file.

        Returns:
            List[str]: A list of file paths of the audio chunks.
        """
        print(f"Chunking audio: {file_path}")
        try:
            audio = AudioSegment.from_file(file_path)
            # audio: PyDub AudioSegment object
            if self.chunk_size == 0 or len(audio) <= self.chunk_size:
                print("No chunking needed for the audio.")
                return [file_path]

            chunks = [audio[i:i + self.chunk_size] for i in range(0, len(audio), self.chunk_size)]
            # chunks: list of PyDub AudioSegment objects, each representing a chunk
            chunk_paths = []

            def export_chunk(idx, chunk):
                chunk_path = os.path.join(self.temp_dir, f'chunk_{idx}.mp3')
                chunk.export(chunk_path, format='mp3')
                print(f"Chunk created: {chunk_path}")
                return chunk_path

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(export_chunk, idx, chunk) for idx, chunk in enumerate(chunks)]
                for future in futures:
                    chunk_paths.append(future.result())

            print(f"{len(chunk_paths)} chunks created.")
            # chunk_paths: list of strings, each representing the path to a chunk file
            return chunk_paths
        except Exception as e:
            print(f"Error chunking audio: {e}")
            raise

    def transcribe(self, input_path: str, method: str, diarization: bool = False) -> str:
        """
        Transcribes audio from a given input path using the specified method, with optional diarization.

        Args:
            input_path (str): The path/url to the audio file or URL.
            method (str): The transcription method to use ('insanely-fast-whisper', 'groq', 'whisper-turbo').
            diarization (bool): Whether to perform speaker diarization. Only applicable for 'groq' method.

        Returns:
            str: The transcribed text with speaker labels.

        Raises:
            ValueError: If the input file format or transcription method is not supported.
        """
        print(f"Starting transcription for: {input_path} with method: {method}, diarization: {diarization}")
        try:
            if input_path.startswith("http://") or input_path.startswith("https://"):
                audio_path = self.download_audio(input_path, self.temp_dir)
                if not audio_path:
                    print(f"Download failed for URL: {input_path}")
                    return ""

            elif input_path.lower().endswith(('.wav', '.mp3', '.m4a')):
                audio_path = input_path
                print(f"Local audio file used: {audio_path}")
            else:
                raise ValueError("Unsupported file format.")

            chunks = self.chunk_audio(audio_path)
            # chunks: list of strings representing paths to audio chunks
            print(f"Transcribing {len(chunks)} chunks.")

            
            if method == 'groq' and diarization:
                # Run diarization on the full audio before transcription
                auth_token = os.getenv('HUGGINGFACE_TOKEN')
                diarization_result = run_diarization(audio_path, auth_token)
                # diarization_result: output from pyannote.audio diarization pipeline
                print("Diarization completed on the full audio.")

            if method == 'insanely-fast-whisper':
                texts = self.transcribe_insanely_fast_whisper(chunks)
                texts = " ".join(texts)
            elif method == 'groq':
                texts = self.transcribe_groq(chunks, audio_path, diarization_result if diarization else None) # Pass full audio path and diarization result
            elif method == 'whisper-turbo':
                texts = self.transcribe_turbo(chunks)
                texts = " ".join(texts)
            else:
                raise ValueError("Unsupported transcription method.")

            return texts
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
        finally:
            self.cleanup()

    def transcribe_whisper(self, chunks: List[str]) -> List[str]:
        """
        Transcribes audio chunks using the Whisper model.

        Args:
            chunks (List[str]): A list of paths to audio chunks.

        Returns:
            List[str]: A list of transcribed text from each chunk.
        """
        from faster_whisper import WhisperModel
        print("Loading Whisper model...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "float32"
            model = WhisperModel("large-v3", device=device, compute_type=compute_type)
            # model: faster_whisper.WhisperModel object
            print(f"Whisper model loaded on device: {device}")

            texts = []
            for idx, chunk in enumerate(chunks, start=1):
                print(f"Transcribing chunk {idx}/{len(chunks)}: {chunk}")
                segments, _ = model.transcribe(chunk, beam_size=5)
                # segments: iterable of segment objects from faster_whisper
                for segment in segments:
                    texts.append(segment.text)
            print("Transcription with Whisper completed.")
            return texts
        except Exception as e:
            print(f"Error during transcription with Whisper: {e}")
            return []

    def transcribe_groq(self, chunks: List[str], audio_path: str, diarization_result) -> str:
        """
        Transcribes audio chunks using the Groq API, integrates diarization information, and formats the output.

        Args:
            chunks (List[str]): A list of paths to audio chunks.
            audio_path (str): The path to the full audio file.
            diarization_result: The diarization result from pyannote.

        Returns:
            str: The transcribed text with speaker labels.
        """
        print("Initializing Groq client for transcription...")

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        chunk_offset = 0  # Keep track of the offset for each chunk
        all_transcriptions = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self._transcribe_chunk_groq, client, chunk, chunk_offset)
                futures.append((future, chunk_offset))  # Store the future along with the chunk offset
                chunk_offset += self.chunk_size / 1000

            for future, chunk_offset in tqdm(futures, desc="Transcription with Groq"):
                transcribed_chunk_with_timestamps = future.result()
                if transcribed_chunk_with_timestamps:
                    all_transcriptions.extend(transcribed_chunk_with_timestamps)  # Use extend instead of append

        # Sort all transcriptions by their start time
        all_transcriptions.sort(key=lambda x: x['start'])

        # Assign speakers based on the diarization result
        final_transcript = self.assign_speakers(all_transcriptions, diarization_result)

        print("Transcription with Groq and diarization integration completed.")
        return final_transcript

    def _transcribe_chunk_groq(self, client, chunk: str, chunk_offset: float) -> list:
        """
        Transcribes a single audio chunk using the Groq API and returns transcription segments with adjusted timestamps.

        Args:
            client: The Groq API client.
            chunk (str): Path to the audio chunk.
            chunk_offset (float): The time offset of the chunk in the full audio.

        Returns:
            list: A list of dictionaries, each containing the start time, end time, and transcribed text of a segment.
        """
        try:
            # Downsample audio to 16000 Hz mono
            audio = AudioSegment.from_file(chunk)
            audio = audio.set_frame_rate(16000).set_channels(1)
            downsampled_chunk_path = os.path.join(self.temp_dir, f'downsampled_{os.path.basename(chunk)}')
            audio.export(downsampled_chunk_path, format='mp3')

            fragment_size_mb = os.path.getsize(downsampled_chunk_path) / (1024 * 1024)
            print(f"Fragment size: {fragment_size_mb:.2f} MB")

            with open(downsampled_chunk_path, "rb") as file:
                data = file.read()
                # data: bytes object representing the audio file content

            translation = client.audio.transcriptions.create(
                file=(os.path.basename(downsampled_chunk_path), data),
                model='whisper-large-v3-turbo',
                response_format="verbose_json",
            )
            # translation: response object from the Groq API
            print("Transcript obtained from Groq:", translation)

            if translation and hasattr(translation, 'text') and hasattr(translation, 'segments'):
                print(f"Chunk transcribed with timestamps: {chunk}")
                
                
                segments_with_timestamps = []
                for segment in translation.segments:
                    start_time = segment['start'] + chunk_offset
                    end_time = segment['end'] + chunk_offset
                    text = segment['text']
                    segments_with_timestamps.append({'start': start_time, 'end': end_time, 'text': text.strip()})
                
                # segments_with_timestamps: list of dictionaries, each with 'start', 'end', 'text' keys
                return segments_with_timestamps

            else:
                print(f"No transcription obtained for chunk: {chunk}")
                return []

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit error encountered: {e}")
                # Attempt to switch to a different model as a fallback
                try:
                    translation = client.audio.transcriptions.create(
                        file=(os.path.basename(downsampled_chunk_path), data),
                        model='whisper-large-v3',
                        response_format="verbose_json",
                    )
                    if translation and hasattr(translation, 'text') and hasattr(translation, 'segments'):
                        print(f"Chunk transcribed with fallback model and timestamps: {chunk}")
                        
                        segments_with_timestamps = []
                        for segment in translation.segments:
                            start_time = segment['start'] + chunk_offset
                            end_time = segment['end'] + chunk_offset
                            text = segment['text']
                            segments_with_timestamps.append({'start': start_time, 'end': end_time, 'text': text.strip()})
                        
                        return segments_with_timestamps
                    
                    else:
                        print(f"No transcription obtained for chunk with fallback model: {chunk}")
                        return []
                except Exception as e:
                    print(f"Error transcribing chunk {chunk} with fallback model: {e}")
                    return []
            else:
                print(f"Error transcribing chunk {chunk} with Groq: {e}")
                return []
        except Exception as e:
            print(f"Error transcribing chunk {chunk} with Groq: {e}")
            return []

    def assign_speakers(self, transcription_segments, diarization_result):
        """
        Assigns speakers to transcription segments based on diarization results, handling edge cases.

        Args:
            transcription_segments (list): A list of dictionaries, where each dictionary contains 
                                            start, end, and text of transcription segments.
            diarization_result: The diarization result from pyannote for the entire audio file.

        Returns:
            str: The final transcribed text formatted with speaker labels.
        """
        # Create a list of diarization segments with speaker labels
        diarization_segments = []
        if diarization_result:
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                diarization_segments.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})

        # Sort diarization segments by start time
        diarization_segments.sort(key=lambda x: x['start'])

        # Assign speakers to transcription segments
        result = []
        diar_idx = 0
        for trans_seg in transcription_segments:
            assigned_speaker = "UNKNOWN"  # Default speaker
            
            # Find the corresponding diarization segment
            while diar_idx < len(diarization_segments):
                diar_seg = diarization_segments[diar_idx]
                
                if diar_seg['start'] <= trans_seg['start'] < diar_seg['end']:
                    assigned_speaker = diar_seg['speaker']
                    break
                elif trans_seg['start'] >= diar_seg['end']:
                    diar_idx += 1
                else:
                    break  # No overlap
            
            # Handle edge cases: 
            # 1. If the transcription segment starts before the first diarization segment
            # 2. Or if it falls between two diarization segments, assign the speaker of the previous diarization segment (if available)
            if assigned_speaker == "UNKNOWN" and diar_idx > 0:
                assigned_speaker = diarization_segments[diar_idx - 1]['speaker']

            result.append({'speaker': assigned_speaker, 'text': trans_seg['text']})

        # Format the output
        formatted_output = ""
        current_speaker = None
        for segment in result:
            if segment['speaker'] != current_speaker:
                current_speaker = segment['speaker']
                formatted_output += f"\n[SPEAKER_{current_speaker}] "
            formatted_output += f"{segment['text']} "

        return formatted_output.strip()
    
    def format_diarization(self, diarization_result):
        """
        Formats the diarization result into a readable string.

        Args:
            diarization_result: The diarization result from pyannote.

        Returns:
            str: A formatted string representing the diarization result.
        """
        diarization_output = ""
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            diarization_output += f"[{turn.start:.2f}-{turn.end:.2f}] SPEAKER_{speaker} "
        return diarization_output

    @lru_cache(maxsize=None)
    def load_turbo_model(self):
        """
        Loads the Whisper Turbo model for transcription.

        Returns:
            tuple: A tuple containing the loaded model and processor.
        """
        print("Loading Whisper Turbo model...")
        try:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_id = "openai/whisper-large-v3"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, device_map='auto'
            )
            processor = AutoProcessor.from_pretrained(model_id)
            # model: transformers.AutoModelForSpeechSeq2Seq object
            # processor: transformers.AutoProcessor object
            print("Whisper Turbo model loaded successfully.")
            return model, processor
        except Exception as e:
            print(f"Error loading Whisper Turbo model: {e}")
            raise

    def transcribe_insanely_fast_whisper(self, chunks: List[str]) -> List[str]:
        """
        Transcribes audio chunks using Insanely Fast Whisper.

        Args:
            chunks (List[str]): A list of paths to audio chunks.

        Returns:
            List[str]: A list of transcribed text from each chunk.
        """
        import subprocess
        import json
        print("Transcription with Insanely Fast Whisper...")
        try:
            texts = []
            for idx, chunk in enumerate(chunks, start=1):
                print(f"Transcribing chunk {idx}/{len(chunks)}: {chunk}")
                command = [
                    "insanely-fast-whisper",
                    "--model-name", "openai/whisper-large-v3",
                    "--file-name", chunk,
                    "--flash", "FLASH",
                    "--hf-token", os.getenv("HUGGINGFACE_TOKEN"),
                    "--transcript-path", os.path.join(self.temp_dir, f"transcript_{idx}.json"),
                    "--batch-size", str(self.batch_size),
                ]
                subprocess.run(command, check=True)
                with open(os.path.join(self.temp_dir, f"transcript_{idx}.json"), "r", encoding="utf-8") as file:
                    transcript = json.load(file)
                    # transcript: dictionary containing the transcription results from insanely-fast-whisper
                    texts.append(transcript["text"])
            print("Transcription with Insanely Fast Whisper completed.")
            return texts
        except Exception as e:
            print(f"Error during transcription with Insanely Fast Whisper: {e}")
            return []

    def transcribe_turbo(self, chunks: List[str]) -> List[str]:
        """
        Transcribes audio chunks using the Whisper Turbo model via a pipeline.

        Args:
            chunks (List[str]): A list of paths to audio chunks.

        Returns:
            List[str]: A list of transcribed text from each chunk.
        """
        print("Initializing transcription with Whisper Turbo...")
        try:
            model, processor = self.load_turbo_model()
            processor.feature_extractor.return_attention_mask = True
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
                return_timestamps=True,
                batch_size=self.batch_size,
                padding=True
            )
            # pipe: transformers.pipeline object for automatic speech recognition
            print(f"Whisper Turbo pipeline configured on device: {'GPU' if device == 0 else 'CPU'}")

            texts = []
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            for idx, batch in enumerate(tqdm(batches, desc="Transcription with Whisper Turbo"), start=1):
                print(f"Transcribing batch {idx}/{len(batches)}")
                results = pipe(batch)
                # results: list of dictionaries containing transcription results from the pipeline
                if isinstance(results, list):
                    for result in results:
                        texts.append(result["text"])
                else:
                    texts.append(results["text"])
            print("Transcription with Whisper Turbo completed.")
            return texts
        except Exception as e:
            print(f"Error during transcription with Whisper Turbo: {e}")
            return []

    def cleanup(self):
        """
        Cleans up the temporary directory used for storing audio chunks.
        """
        print("Cleaning up temporary directory...")
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print("Temporary directory cleaned up successfully.")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

def run_diarization(audio_path, auth_token):
    """
    Performs speaker diarization on an audio file using pyannote.audio.

    Args:
        audio_path (str): Path to the audio file.
        auth_token (str): Authentication token for Hugging Face.

    Returns:
        The diarization result, or None if an error occurs.
    """
    try:
        # Initialize pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        # pipeline: pyannote.audio Pipeline object

        # Setup GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("FATAL ISSUE: No GPU available for diarization !")
        pipeline.to(device)

        # Load and validate audio
        waveform, sample_rate = torchaudio.load(audio_path)
        print("CURRENT SAMPLE RATE:", sample_rate)
        # waveform: torch.Tensor representing the audio waveform
        # sample_rate: integer representing the sample rate of the audio
        
        # Run diarization with progress hook
        with ProgressHook() as hook:
            diarization = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
                min_speakers=1,
                max_speakers=2
            )
            # diarization: pyannote.core.Annotation object representing the diarization result

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
    transcriber = YouTubeTranscriber(chunk_size=30, batch_size=1, language='en')
    input_path = "https://youtu.be/QbelS64US98?si=HLqtoEnkNUYN85Vx"
    method = "groq"
    diarization = True  # Set to True to enable diarization, False to disable
    transcriptions = transcriber.transcribe(input_path, method, diarization)
    print(transcriptions)