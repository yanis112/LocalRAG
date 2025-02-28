import cloudscraper
import re
import subprocess
import os
import shutil
import time
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from m3u8 import M3U8
from tqdm import tqdm
import base64

class EnhancedVideoDownloader:
    def __init__(self, url):
        self.url = url
        self.scraper = cloudscraper.create_scraper()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://papadustream.day/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def extract_m3u8(self):
        for attempt in range(3):
            try:
                response = self.scraper.get(self.url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')

                # Méthode 1: Extraction depuis les balises script avancées
                script_patterns = [
                    r'player\.setup\(({.*?})\);',
                    r'sources:\s*(\[.*?\])',
                    r'file:\s*["\'](.*?\.m3u8.*?)["\']'
                ]
                for pattern in script_patterns:
                    matches = re.findall(pattern, response.text, re.DOTALL)
                    for match in matches:
                        try:
                            if match.startswith('{'):
                                data = json.loads(match)
                                if 'sources' in data:
                                    for source in data['sources']:
                                        if 'file' in source and '.m3u8' in source['file']:
                                            return source['file']
                            elif '.m3u8' in match:
                                return urljoin(self.url, match)
                        except:
                            continue

                # Méthode 2: Analyse des iframes imbriqués
                def parse_iframe(iframe_url):
                    try:
                        iframe_response = self.scraper.get(iframe_url, headers=self.headers)
                        iframe_soup = BeautifulSoup(iframe_response.text, 'html.parser')
                        
                        # Recherche récursive dans les sous-iframes
                        nested_iframe = iframe_soup.find('iframe', {'src': True})
                        if nested_iframe:
                            return parse_iframe(urljoin(iframe_url, nested_iframe['src']))
                            
                        # Détection de sources vidéo
                        source = iframe_soup.find('source', {'src': True})
                        if source and '.m3u8' in source['src']:
                            return urljoin(iframe_url, source['src'])
                            
                        # Détection de scripts dans l'iframe
                        script = iframe_soup.find('script', string=re.compile(r'\.m3u8'))
                        if script:
                            url_match = re.search(r'(https?://[^\s"\']+\.m3u8[^\s"\']*)', script.text)
                            if url_match:
                                return url_match.group(1)
                                
                    except Exception as e:
                        print(f"Erreur iframe: {e}")
                    return None

                iframe = soup.find('iframe', {'src': True})
                if iframe:
                    iframe_url = urljoin(self.url, iframe['src'])
                    m3u8_candidate = parse_iframe(iframe_url)
                    if m3u8_candidate:
                        return m3u8_candidate

                # Méthode 3: Regex amélioré avec décodage d'URL
                encoded_patterns = [
                    r"atob\('([^']+)'\)",
                    r"decodeURIComponent\('([^']+)'\)"
                ]
                for pattern in encoded_patterns:
                    matches = re.findall(pattern, response.text)
                    for match in matches:
                        try:
                            decoded = base64.b64decode(match).decode() if '==' in match else requests.utils.unquote(match)
                            if '.m3u8' in decoded:
                                return decoded
                        except:
                            continue

                # Méthode 4: Recherche exhaustive dans le code HTML
                patterns = [
                    r'(https?://[^\s"\']+\.m3u8[^\s"\']*)',
                    r'file["\']?\s*:\s*["\']((?:https?%3A%2F%2F|//)[^"\']+)["\']',
                    r'var\s+source\s*=\s*["\']([^"\']+\.m3u8[^"\']*)["\']'
                ]
                for pattern in patterns:
                    matches = re.finditer(pattern, response.text, re.IGNORECASE)
                    for match in matches:
                        candidate = urljoin(self.url, match.group(1).replace('%3A%2F%2F', '://'))
                        if self.validate_m3u8(candidate):
                            return candidate

            except Exception as e:
                print(f"Tentative {attempt+1} échouée: {str(e)}")
                time.sleep(2**attempt)
                
        raise Exception("Échec de l'extraction du manifeste m3u8 après 3 tentatives")

    def validate_m3u8(self, url):
        try:
            response = self.scraper.get(url, headers=self.headers, timeout=10, stream=True)
            content_start = response.raw.read(16)
            if content_start.startswith(b'#EXTM3U'):
                return True
            return 'm3u8' in response.headers.get('Content-Type', '')
        except Exception as e:
            print(f"Validation failed: {str(e)}")
            return False

    def download_stream(self, output_filename):
        m3u8_url = self.extract_m3u8()
        print(f"Manifeste trouvé: {m3u8_url}")
        
        # Configuration avancée avec gestion du son
        master_playlist = self.scraper.get(m3u8_url, headers=self.headers).text
        m3u8_obj = M3U8(master_playlist)
        
        # Sélection de la meilleure qualité avec audio
        best_stream = None
        max_bandwidth = 0
        for playlist in m3u8_obj.playlists:
            if playlist.stream_info.bandwidth > max_bandwidth and playlist.stream_info.audio:
                best_stream = playlist
                max_bandwidth = playlist.stream_info.bandwidth

        if not best_stream:
            raise Exception("Aucun flux vidéo valide trouvé")

        # Téléchargement parallèle avec gestion des erreurs
        temp_dir = 'temp_segments'
        os.makedirs(temp_dir, exist_ok=True)
        
        base_uri = urljoin(m3u8_url, best_stream.uri)
        segment_base = base_uri.rsplit('/', 1)[0] + '/'

        segments = best_stream.media_segment
        print(f"Téléchargement de {len(segments)} segments...")
        
        for i, segment in enumerate(tqdm(segments)):
            segment_url = urljoin(segment_base, segment.uri)
            for retry in range(3):
                try:
                    response = self.scraper.get(segment_url, headers={
                        **self.headers,
                        'Origin': urlparse(m3u8_url).scheme + '://' + urlparse(m3u8_url).netloc,
                        'Referer': self.url
                    }, timeout=15)
                    with open(f'{temp_dir}/segment_{i:04d}.ts', 'wb') as f:
                        f.write(response.content)
                    break
                except Exception as e:
                    if retry == 2:
                        raise
                    time.sleep(1)

        # Assemblage avec FFmpeg
        subprocess.run([
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', f'{temp_dir}/segment_%04d.ts',
            '-c', 'copy',
            '-movflags', 'faststart',
            '-bsf:a', 'aac_adtstoasc',
            output_filename
        ], check=True)

        shutil.rmtree(temp_dir)
        print(f"Téléchargement terminé: {output_filename}")

if __name__ == '__main__':
    downloader = EnhancedVideoDownloader(
        'https://papadustream.day/categorie-series/drame-s/10697-game-of-thrones-house-of-the-dragon/1-saison-l/8-episode.html'
    )
    downloader.download_stream('episode_8.mp4')