import os
import csv
from src.transcription_utils import YouTubeTranscriber
from tqdm import tqdm
from typing import List
from urllib.parse import urlparse, parse_qs
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def read_urls(file_path: str) -> List[str]:
    logging.info(f"Lecture des URLs depuis le fichier: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        urls = [row[0].strip() for row in reader if row]
    logging.info(f"{len(urls)} URLs trouvées.")
    return urls

def read_transcribed_log(log_path: str) -> List[str]:
    if not os.path.exists(log_path):
        logging.info(f"Aucun log de transcription trouvé à {log_path}.")
        return []
    with open(log_path, 'r', encoding='utf-8') as file:
        transcribed = file.read().splitlines()
    logging.info(f"{len(transcribed)} URLs déjà transcrites.")
    return transcribed

def log_transcribed_url(log_path: str, url: str):
    with open(log_path, 'a', encoding='utf-8') as file:
        file.write(url + '\n')
    logging.info(f"URL transcrite ajoutée au log: {url}")

def get_video_id(url: str) -> str:
    """
    Extrait l'ID de la vidéo à partir d'une URL YouTube.
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    else:
        return None

def save_transcription(output_dir: str, url: str, transcription: List[str]):
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError(f"Impossible d'extraire l'ID de la vidéo depuis l'URL: {url}")
    output_path = os.path.join(output_dir, f"{video_id}.txt")
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(transcription))
    logging.info(f"Transcription sauvegardée dans: {output_path}")

def main():
    urls_file = 'video_urls.csv'
    log_file = 'transcribed_log.txt'
    output_dir = 'data/politique'
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Répertoire de sortie vérifié ou créé: {output_dir}")

    urls = read_urls(urls_file)
    transcribed_urls = read_transcribed_log(log_file)

    transcriber = YouTubeTranscriber(chunk_size=1500, batch_size=1, language='fr')

    for url in tqdm(urls, desc="Transcribing videos"):
        logging.info(f"Début de la transcription pour l'URL: {url}")
        if url in transcribed_urls:
            logging.info(f"URL déjà transcrite, passage à la suivante: {url}")
            continue

        try:
            transcription = transcriber.transcribe(url, 'groq')
            if transcription:
                save_transcription(output_dir, url, transcription)
                log_transcribed_url(log_file, url)
                logging.info(f"Transcription réussie pour l'URL: {url}")
            else:
                logging.warning(f"Aucune transcription obtenue pour l'URL: {url}")
        except Exception as e:
            logging.error(f"Erreur lors de la transcription de l'URL {url}: {e}")

if __name__ == "__main__":
    main()
