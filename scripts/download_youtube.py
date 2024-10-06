import os
import csv
import asyncio
from src.transcription_utils import YouTubeTranscriber
from tqdm import tqdm
from typing import List

def read_urls(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]

def read_transcribed_log(log_path: str) -> List[str]:
    if not os.path.exists(log_path):
        return []
    with open(log_path, 'r') as file:
        return file.read().splitlines()

def log_transcribed_url(log_path: str, url: str):
    with open(log_path, 'a') as file:
        file.write(url + '\n')

def save_transcription(output_dir: str, url: str, transcription: List[str]):
    video_id = url.split('=')[-1]
    output_path = os.path.join(output_dir, f"{video_id}.txt")
    with open(output_path, 'w') as file:
        file.write('\n'.join(transcription))

async def main():
    urls_file = 'video_urls.csv'
    log_file = 'transcribed_log.txt'
    output_dir = 'data/politique'
    os.makedirs(output_dir, exist_ok=True)

    urls = read_urls(urls_file)
    transcribed_urls = read_transcribed_log(log_file)

    transcriber = YouTubeTranscriber(chunk_size=100, batch_size=1, language='fr')

    for url in tqdm(urls, desc="Transcribing videos"):
        if url in transcribed_urls:
            print(f"Skipping already transcribed URL: {url}")
            continue

        try:
            transcription = await transcriber.transcribe(url, 'groq')
            save_transcription(output_dir, url, transcription)
            log_transcribed_url(log_file, url)
            print(f"Successfully transcribed and saved URL: {url}")
        except Exception as e:
            print(f"Error transcribing URL {url}: {e}")

if __name__ == "__main__":
    if not asyncio.get_event_loop().is_running():
        asyncio.run(main())
    else:
        asyncio.create_task(main())