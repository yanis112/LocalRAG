from suno import Suno, ModelVersions
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class SunoGenerator:
    def __init__(self, model_version=ModelVersions.CHIRP_V3_5):
        self.client = Suno(
            cookie=os.getenv('SUNO_COOKIE'),
            model_version=model_version
        )

    def generate_song(self, prompt, is_custom, tags=None, title=None, make_instrumental=None, wait_audio=True):
        return self.client.generate(
            prompt=prompt,
            is_custom=is_custom,
            tags=tags,
            title=title,
            make_instrumental=make_instrumental,
            wait_audio=wait_audio
        )

    def download_songs(self, songs):
        for song in songs:
            file_path = self.client.download(song=song)
            print(f"Song downloaded to: {file_path}")

# Example usage
if __name__ == "__main__":
    generator = SunoGenerator()

    # Generate a song with all parameters
    songs = generator.generate_song(
        prompt="a dark techno track: 126 BPM, thunderous sub-bass, minimal percussion, eerie synths. Club-ready, bass-centric mix.",
        #tags="English men voice",
        title="Hard Techno",
        make_instrumental=True,
        is_custom=False,
        wait_audio=True
    )

    # Download generated songs
    generator.download_songs(songs)