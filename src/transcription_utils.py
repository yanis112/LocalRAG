import os
import shutil
import torch
import yt_dlp
from pydub import AudioSegment
from groq import Groq
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List
from functools import lru_cache
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class YouTubeTranscriber:
    def __init__(self, chunk_size: int = 0, batch_size: int = 1, language: str = 'en'):
        self.chunk_size = chunk_size * 1000  # Convertir en millisecondes
        self.batch_size = batch_size
        self.language = language
        self.temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        logging.info(f"Répertoire temporaire vérifié ou créé: {self.temp_dir}")

    def download_audio(self, url: str, output_dir: str) -> str:
        logging.info(f"Téléchargement de l'audio depuis l'URL: {url}")
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
                filename = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            logging.info(f"Audio téléchargé: {filename}")
            return filename
        except yt_dlp.utils.DownloadError as e:
            logging.error(f"Erreur lors du téléchargement de l'audio: {e}")
            return None

    def chunk_audio(self, file_path: str) -> List[str]:
        logging.info(f"Découpage de l'audio: {file_path}")
        try:
            audio = AudioSegment.from_file(file_path)
            if self.chunk_size == 0 or len(audio) <= self.chunk_size:
                logging.info("Aucun découpage nécessaire pour l'audio.")
                return [file_path]
            chunks = [audio[i:i + self.chunk_size] for i in range(0, len(audio), self.chunk_size)]
            chunk_paths = []
            for idx, chunk in enumerate(chunks):
                chunk_path = os.path.join(self.temp_dir, f'chunk_{idx}.mp3')
                chunk.export(chunk_path, format='mp3')
                logging.info(f"Chunk créé: {chunk_path}")
                chunk_paths.append(chunk_path)
            logging.info(f"{len(chunk_paths)} chunks créés.")
            return chunk_paths
        except Exception as e:
            logging.error(f"Erreur lors du découpage de l'audio: {e}")
            raise

    def transcribe(self, input_path: str, method: str) -> List[str]:
        logging.info(f"Début de la transcription pour: {input_path} avec la méthode: {method}")
        try:
            if input_path.startswith("http://") or input_path.startswith("https://"):
                audio_path = self.download_audio(input_path, self.temp_dir)
                if not audio_path:
                    logging.warning(f"Téléchargement échoué pour l'URL: {input_path}")
                    return []
            elif input_path.lower().endswith(('.wav', '.mp3', '.m4a')):
                audio_path = input_path
                logging.info(f"Fichier audio local utilisé: {audio_path}")
            else:
                raise ValueError("Format de fichier non supporté.")
            
            chunks = self.chunk_audio(audio_path)
            logging.info(f"Transcription des {len(chunks)} chunks.")

            if method == 'insanely-fast-whisper':
                texts = self.transcribe_whisper(chunks)
            elif method == 'groq':
                texts = self.transcribe_groq(chunks)
            elif method == 'whisper-turbo':
                texts = self.transcribe_turbo(chunks)
            else:
                raise ValueError("Méthode de transcription non supportée.")
            return texts
        except Exception as e:
            logging.error(f"Erreur lors de la transcription: {e}")
            return []
        finally:
            self.cleanup()

    def transcribe_whisper(self, chunks: List[str]) -> List[str]:
        from faster_whisper import WhisperModel
        logging.info("Chargement du modèle Whisper...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "float32"
            model = WhisperModel("large-v3", device=device, compute_type=compute_type)
            logging.info(f"Modèle Whisper chargé sur le device: {device}")

            texts = []
            for idx, chunk in enumerate(chunks, start=1):
                logging.info(f"Transcription du chunk {idx}/{len(chunks)}: {chunk}")
                segments, _ = model.transcribe(chunk, beam_size=5)
                for segment in segments:
                    texts.append(segment.text)
            logging.info("Transcription avec Whisper terminée.")
            return texts
        except Exception as e:
            logging.error(f"Erreur lors de la transcription avec Whisper: {e}")
            return []

    def transcribe_groq(self, chunks: List[str]) -> List[str]:
        logging.info("Initialisation du client Groq pour la transcription...")
        texts = []
        client = Groq()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._transcribe_chunk_groq, client, chunk) for chunk in chunks]
            for future in tqdm(futures, desc="Transcription avec Groq"):
                result = future.result()
                if result:
                    texts.append(result)
        logging.info("Transcription avec Groq terminée.")
        return texts

    def _transcribe_chunk_groq(self, client, chunk: str) -> str:
        try:
            with open(chunk, "rb") as file:
                data = file.read()
            translation = client.audio.transcriptions.create(
                file=(os.path.basename(chunk), data),
                model="whisper-large-v3",
                response_format="json",
                language=self.language,
                temperature=0.0
            )
            if translation and hasattr(translation, 'text'):
                logging.info(f"Chunk transcrit avec succès: {chunk}")
                return translation.text
            else:
                logging.warning(f"Aucune transcription obtenue pour le chunk: {chunk}")
                return None
        except Exception as e:
            logging.error(f"Erreur lors de la transcription du chunk {chunk} avec Groq: {e}")
            return None

    @lru_cache(maxsize=None)
    def load_turbo_model(self):
        logging.info("Chargement du modèle Whisper Turbo...")
        try:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_id = "openai/whisper-large-v3-turbo"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, device_map='auto'
            )
            processor = AutoProcessor.from_pretrained(model_id)
            logging.info("Modèle Whisper Turbo chargé avec succès.")
            return model, processor
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle Whisper Turbo: {e}")
            raise

    def transcribe_turbo(self, chunks: List[str]) -> List[str]:
        logging.info("Initialisation de la transcription avec Whisper Turbo...")
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
            logging.info(f"Pipeline Whisper Turbo configuré sur le device: {'GPU' if device == 0 else 'CPU'}")

            texts = []
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            for idx, batch in enumerate(tqdm(batches, desc="Transcription avec Whisper Turbo"), start=1):
                logging.info(f"Transcription du batch {idx}/{len(batches)}")
                results = pipe(batch)
                if isinstance(results, list):
                    for result in results:
                        texts.append(result["text"])
                else:
                    texts.append(results["text"])
            logging.info("Transcription avec Whisper Turbo terminée.")
            return texts
        except Exception as e:
            logging.error(f"Erreur lors de la transcription avec Whisper Turbo: {e}")
            return []

    def cleanup(self):
        logging.info("Nettoyage du répertoire temporaire...")
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logging.info("Répertoire temporaire nettoyé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors du nettoyage du répertoire temporaire: {e}")

if __name__ == "__main__":
    transcriber = YouTubeTranscriber(chunk_size=120, batch_size=5, language='en')
    input_path = "https://youtu.be/lIfbv3winLo?si=VuZEKbB94VvlA7PQ"
    method = "groq"
    transcriptions = transcriber.transcribe(input_path, method)
    for text in transcriptions:
        print(text)