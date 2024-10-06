import os
import shutil
import torch
import tempfile
import yt_dlp
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from groq import Groq
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List
from functools import lru_cache
import asyncio
from dotenv import load_dotenv
import time

load_dotenv()

class YouTubeTranscriber:
    def __init__(self, chunk_size: int = 0, batch_size: int = 1, language: str = 'en'):
        self.chunk_size = chunk_size * 1000  # Convert to milliseconds
        self.batch_size = batch_size
        self.language = language

    def download_video(self, url: str, output_dir: str) -> str:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return filename
        except yt_dlp.utils.DownloadError as e:
            print(f"Error downloading video: {e}")
            return None

    def extract_audio_from_video(self, file_path: str) -> str:
        video = VideoFileClip(file_path)
        audio_file_path = file_path.rsplit('.', 1)[0] + '.wav'
        video.audio.write_audiofile(audio_file_path)
        return audio_file_path

    def chunk_audio(self, file_path: str) -> List[str]:
        audio = AudioSegment.from_file(file_path)
        if self.chunk_size == 0:
            return [file_path]
        chunks = [audio[i:i + self.chunk_size] for i in range(0, len(audio), self.chunk_size)]
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        chunk_paths = []
        for idx, chunk in enumerate(chunks):
            chunk_path = os.path.join(temp_dir, f'chunk_{idx}.wav')
            chunk.export(chunk_path, format='wav')
            time.sleep(0.1)  # Add a small delay to ensure the file is fully written
            chunk_paths.append(chunk_path)
        return chunk_paths

    async def transcribe(self, input_path: str, method: str):
        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        try:
            if input_path.startswith("http://") or input_path.startswith("https://"):
                audio_path = self.download_video(input_path, temp_dir)
                if not audio_path:
                    return []
            elif input_path.lower().endswith(('.wav', '.mp3', '.m4a')):
                audio_path = input_path
            else:
                audio_path = self.extract_audio_from_video(input_path)
            chunks = self.chunk_audio(audio_path)

            if method == 'insanely-fast-whisper':
                texts = self.transcribe_whisper(chunks)
            elif method == 'groq':
                texts = await self.transcribe_groq(chunks)
            elif method == 'whisper-turbo':
                texts = self.transcribe_turbo(chunks)
            else:
                raise ValueError("Unsupported transcription method.")
        finally:
            pass  # Supprimez ou commentez la ligne de suppression ici

        # Déplacez la suppression du dossier temporaire ici
        shutil.rmtree(temp_dir)
        return texts


    def transcribe_whisper(self, chunks: List[str]) -> List[str]:
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)

        texts = []
        for chunk in chunks:
            segments, _ = model.transcribe(chunk, beam_size=5)
            for segment in segments:
                texts.append(segment.text)
        return texts

    async def transcribe_groq(self, chunks: List[str]) -> List[str]:
        client = Groq()
        texts = []

        async def transcribe_chunk(chunk):
            with open(chunk, "rb") as file:
                data = file.read()
            translation = await asyncio.to_thread(
                client.audio.transcriptions.create,
                file=(chunk, data),
                model="whisper-large-v3",
                response_format="json",
                language=self.language,
                temperature=0.0
            )
            return translation.text

        tasks = [transcribe_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        texts.extend(results)
        return texts

    @lru_cache(maxsize=1)
    def load_turbo_model(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor

    def transcribe_turbo(self, chunks: List[str]) -> List[str]:
        model, processor = self.load_turbo_model()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        texts = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            results = pipe(batch, batch_size=self.batch_size)
            if isinstance(results, list):
                for result in results:
                    texts.append(result["text"])
            else:
                texts.append(results["text"])
        return texts

if __name__ == "__main__":
    transcriber = YouTubeTranscriber(chunk_size=120, batch_size=5, language='en')
    input_path = "https://youtu.be/lIfbv3winLo?si=VuZEKbB94VvlA7PQ"  # or a local file path
    method = "groq"
    asyncio.run(transcriber.transcribe(input_path, method))