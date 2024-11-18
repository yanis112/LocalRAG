# video_subtitler.py

import moviepy.editor as mp
import srt
import re
import os
from moviepy import config as mpc
from moviepy.video.VideoClip import ColorClip

# Spécifiez le chemin vers magick.exe d'ImageMagick
# Remplacez le chemin ci-dessous par le chemin réel de votre installation
mpc.IMAGEMAGICK_BINARY = "C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"

# Optionnellement, assurez-vous que le répertoire est dans le PATH
os.environ["PATH"] += os.pathsep + "C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/"

os.environ["IMAGEMAGICK_BINARY"] = "C:/Program Files/ImageMagick-7.1.1-Q16-HDRI"

class VideoSubtitler:
    """
    Une classe pour superposer des sous-titres sur une vidéo avec des mots mis en évidence au fur et à mesure qu'ils sont prononcés.
    """

    def __init__(self, video_path, srt_path, output_path='output_video.mp4', font='Arial', font_size=24,
                 color='white', highlight_color='yellow', stroke_color='black', stroke_width=1.5,
                 max_chars=30, max_duration=2.5, max_gap=1.5):
        """
        Initialise le VideoSubtitler avec les chemins et charge la vidéo.

        Args:
            video_path (str): Chemin vers le fichier vidéo d'entrée.
            srt_path (str): Chemin vers le fichier .srt de sous-titres.
            output_path (str, optional): Chemin pour enregistrer le fichier vidéo de sortie. Défaut 'output_video.mp4'.
            font (str, optional): Police utilisée pour les sous-titres. Défaut 'Arial'.
            font_size (int, optional): Taille de police pour les sous-titres. Défaut 24.
            color (str, optional): Couleur du texte des sous-titres. Défaut 'white'.
            highlight_color (str, optional): Couleur de mise en évidence des mots. Défaut 'yellow'.
            stroke_color (str, optional): Couleur du contour du texte. Défaut 'black'.
            stroke_width (float, optional): Largeur du contour du texte. Défaut 1.5.
            max_chars (int, optional): Nombre maximum de caractères par ligne. Défaut 30.
            max_duration (float, optional): Durée maximale d'une ligne en secondes. Défaut 2.5.
            max_gap (float, optional): Durée maximale de silence pour déclencher une nouvelle ligne. Défaut 1.5.
        """
        self.video_path = video_path
        self.srt_path = srt_path
        self.output_path = output_path
        self.font = font
        self.font_size = font_size
        self.color = color
        self.highlight_color = highlight_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.max_chars = max_chars
        self.max_duration = max_duration
        self.max_gap = max_gap

        # Charge le clip vidéo
        self.video_clip = mp.VideoFileClip(self.video_path)

        # Initialise les sous-titres et les clips de sous-titres
        self.subtitles = []
        self.word_level_subtitles = []
        self.line_level_subtitles = []
        self.subtitle_clips = []

    def parse_srt(self):
        """
        Analyse le fichier de sous-titres .srt et stocke les sous-titres dans self.subtitles.
        """
        with open(self.srt_path, 'r', encoding='utf-8') as srt_file:
            srt_content = srt_file.read()
            self.subtitles = list(srt.parse(srt_content))

    def subtitles_to_word_level(self):
        """
        Convertit les sous-titres en informations au niveau des mots.
        """
        for subtitle in self.subtitles:
            content = subtitle.content.strip()
            words = re.findall(r'\w+|[^\w\s]', content, re.UNICODE)
            num_words = len(words)
            if num_words == 0:
                continue
            start_time = subtitle.start.total_seconds()
            end_time = subtitle.end.total_seconds()
            duration = end_time - start_time
            word_duration = duration / num_words
            word_level_info = []
            current_time = start_time
            for word in words:
                word_info = {
                    "word": word,
                    "start": current_time,
                    "end": current_time + word_duration
                }
                word_level_info.append(word_info)
                current_time += word_duration
            self.word_level_subtitles.extend(word_level_info)

    def split_text_into_lines(self):
        """
        Divise le texte en lignes basées sur max_chars, max_duration et max_gap.
        """
        data = self.word_level_subtitles
        subtitles = []
        line = []
        line_duration = 0
        line_chars = 0

        for idx, word_data in enumerate(data):
            word = word_data["word"]
            start = word_data["start"]
            end = word_data["end"]
            duration = end - start

            line.append(word_data)
            line_duration += duration

            temp = " ".join(item["word"] for item in line)
            new_line_chars = len(temp)

            duration_exceeded = line_duration > self.max_duration
            chars_exceeded = new_line_chars > self.max_chars

            if idx > 0:
                gap = word_data['start'] - data[idx - 1]['end']
                maxgap_exceeded = gap > self.max_gap
            else:
                maxgap_exceeded = False

            if duration_exceeded or chars_exceeded or maxgap_exceeded:
                if line:
                    subtitle_line = {
                        "text": " ".join(item["word"] for item in line),
                        "start": line[0]["start"],
                        "end": line[-1]["end"],
                        "words": line.copy()
                    }
                    subtitles.append(subtitle_line)
                    line = []
                    line_duration = 0
                    line_chars = 0

            line_chars = new_line_chars

        if line:
            subtitle_line = {
                "text": " ".join(item["word"] for item in line),
                "start": line[0]["start"],
                "end": line[-1]["end"],
                "words": line.copy()
            }
            subtitles.append(subtitle_line)

        self.line_level_subtitles = subtitles

    def create_caption(self, line_info):
        """
        Crée des clips de sous-titres pour une ligne de texte avec les mots mis en évidence au fur et à mesure qu'ils sont prononcés.
        """
        word_clips = []
        positions = []

        x_pos = 0
        y_pos = 0
        line_width = 0
        frame_width = self.video_clip.w
        frame_height = self.video_clip.h

        x_buffer = frame_width * 0.05  # 5% du cadre
        max_line_width = frame_width - 2 * x_buffer

        fontsize = self.font_size

        space_clip = mp.TextClip(" ", font=self.font, fontsize=fontsize, color=self.color)
        space_width, _ = space_clip.size

        line_start = line_info['start']
        line_end = line_info['end']
        line_duration = line_end - line_start

        # Première passe pour déterminer les positions
        for word_info in line_info['words']:
            word_clip = mp.TextClip(word_info['word'], font=self.font, fontsize=fontsize, color=self.color,
                                    stroke_color=self.stroke_color, stroke_width=self.stroke_width)
            word_width, word_height = word_clip.size

            if line_width + word_width > max_line_width:
                # Passe à la ligne suivante
                x_pos = 0
                y_pos += word_height + 10  # 10 pixels d'espacement vertical
                line_width = 0

            positions.append({
                "x_pos": x_pos + x_buffer,
                "y_pos": y_pos,
                "width": word_width,
                "height": word_height,
                "word": word_info['word'],
                "start": word_info['start'],
                "end": word_info['end'],
                "duration": word_info['end'] - word_info['start']
            })

            x_pos += word_width + space_width
            line_width += word_width + space_width

        # Calcul de la hauteur totale pour centrer verticalement
        total_height = y_pos + word_height
        base_y = frame_height - total_height - 50  # Ajustez pour positionner verticalement

        # Création des clips de mots
        for pos in positions:
            # Mot en couleur normale pour toute la durée de la ligne
            word_clip = mp.TextClip(pos['word'], font=self.font, fontsize=fontsize, color=self.color,
                                    stroke_color=self.stroke_color, stroke_width=self.stroke_width).set_start(line_start).set_duration(line_duration)
            word_clip = word_clip.set_position((pos['x_pos'], base_y + pos['y_pos']))
            word_clips.append(word_clip)

            # Mot mis en évidence pendant sa durée
            highlight_clip = mp.TextClip(pos['word'], font=self.font, fontsize=fontsize, color=self.highlight_color,
                                         stroke_color=self.stroke_color, stroke_width=self.stroke_width).set_start(pos['start']).set_duration(pos['duration'])
            highlight_clip = highlight_clip.set_position((pos['x_pos'], base_y + pos['y_pos']))
            word_clips.append(highlight_clip)

        # Création d'un fond semi-transparent derrière le texte
        max_width = frame_width
        max_height = total_height

        bg_clip = mp.ColorClip(size=(int(max_width), int(max_height + 20)), color=(0, 0, 0))
        bg_clip = bg_clip.set_opacity(0.6)
        bg_clip = bg_clip.set_start(line_start).set_duration(line_duration)
        bg_clip = bg_clip.set_position(('center', base_y - 10))

        # Création du clip composite pour la ligne
        line_clip = mp.CompositeVideoClip([bg_clip] + word_clips)
        self.subtitle_clips.append(line_clip)

    def create_subtitle_clips(self):
        """
        Crée les clips de sous-titres pour toutes les lignes.
        """
        for line_info in self.line_level_subtitles:
            self.create_caption(line_info)

    def overlay_subtitles(self):
        """
        Superpose les clips de sous-titres sur le clip vidéo principal.

        Returns:
            VideoClip: Un CompositeVideoClip avec les sous-titres superposés.
        """
        # Combine le clip vidéo avec tous les clips de sous-titres
        final_clip = mp.CompositeVideoClip([self.video_clip] + self.subtitle_clips)
        return final_clip

    def render_video(self):
        """
        Rendu de la vidéo finale avec les sous-titres et enregistre-la à l'emplacement de sortie.
        """
        final_clip = self.overlay_subtitles()
        # Conserver l'audio original
        final_clip.audio = self.video_clip.audio
        final_clip.write_videofile(self.output_path, codec='libx264', audio_codec='aac')

    def process(self):
        """
        Traite la vidéo en analysant les sous-titres, en créant des clips de sous-titres et en rendant la vidéo.
        """
        print("Analyse des sous-titres...")
        self.parse_srt()
        print("Conversion en informations au niveau des mots...")
        self.subtitles_to_word_level()
        print("Division du texte en lignes...")
        self.split_text_into_lines()
        print("Création des clips de sous-titres...")
        self.create_subtitle_clips()
        print("Rendu de la vidéo finale...")
        self.render_video()
        print(f"Vidéo enregistrée dans {self.output_path}")


if __name__ == '__main__':
    # Exemple d'utilisation
    video_file = "test.mp4"
    subtitle_file = "test.srt"
    output_file = "output_video.mp4"

    # Crée une instance de VideoSubtitler avec une taille de police plus grande et des styles
    subtitler = VideoSubtitler(
        video_file,
        subtitle_file,
        output_path=output_file,
        font='Arial',
        font_size=200,  # Taille de police plus grande
        color='black',
        highlight_color='yellow',
        stroke_color='black',
        stroke_width=2,
        max_chars=30,
        max_duration=2.5,
        max_gap=1.5
    )

    # Traite la vidéo pour ajouter les sous-titres
    subtitler.process()
