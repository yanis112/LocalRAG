import os
import csv
import logging
from typing import List
from tqdm import tqdm
from src.aux_utils.subtitle_scrapper import get_youtube_subtitles,get_video_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def read_urls(file_path: str) -> List[str]:
    logging.info(f"Reading URLs from file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        urls = [row[0].strip() for row in reader if row]
    logging.info(f"Found {len(urls)} URLs.")
    return urls

def read_transcribed_log(log_path: str) -> List[str]:
    if not os.path.exists(log_path):
        logging.info(f"No transcription log found at {log_path}.")
        return []
    with open(log_path, 'r', encoding='utf-8') as file:
        transcribed = file.read().splitlines()
    logging.info(f"{len(transcribed)} URLs already processed.")
    return transcribed

def log_transcribed_url(log_path: str, url: str):
    with open(log_path, 'a', encoding='utf-8') as file:
        file.write(url + '\n')
    logging.info(f"Processed URL added to log: {url}")

def save_subtitles(output_dir: str, url: str, subtitles: str):
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    output_path = os.path.join(output_dir, f"{video_id}.txt")
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(subtitles)
    logging.info(f"Subtitles saved to: {output_path}")

def main(k: int):
    urls_file = 'video_urls.csv'
    log_file = 'transcribed_log.txt'
    output_dir = 'data/politique'
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory checked/created: {output_dir}")

    urls = read_urls(urls_file)[:k]  # Only take the first k URLs
    processed_urls = read_transcribed_log(log_file)

    for url in tqdm(urls, desc="Processing videos"):
        logging.info(f"Starting subtitle extraction for URL: {url}")
        if url in processed_urls:
            logging.info(f"URL already processed, skipping: {url}")
            continue

        try:
            subtitles = get_youtube_subtitles(url, languages=['fr'])
            if subtitles:
                save_subtitles(output_dir, url, subtitles)
                log_transcribed_url(log_file, url)
                logging.info(f"Subtitle extraction successful for URL: {url}")
            else:
                logging.warning(f"No subtitles found for URL: {url}")
        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")

if __name__ == "__main__":
    k = 100  # Change this to the desired number of URLs to process
    main(k)