from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from urllib.parse import urlparse, parse_qs

def get_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    try:
        if 'youtube.com' in url:
            query = parse_qs(urlparse(url).query)
            return query['v'][0]
        elif 'youtu.be' in url:
            path = urlparse(url).path
            return path.strip('/')
        else:
            raise ValueError("Invalid YouTube URL")
    except Exception:
        raise ValueError("Could not extract video ID from URL")

def get_youtube_subtitles(video_url, languages=['en']):
    """Get subtitles from YouTube video, including auto-generated ones"""
    try:
        video_id = get_video_id(video_url)
        print(f"Video ID: {video_id}")
        
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get manual transcripts first
        try:
            transcript = transcript_list.find_transcript(languages)
        except NoTranscriptFound:
            # If no manual transcript, try to get auto-generated ones
            try:
                transcript = transcript_list.find_generated_transcript(languages)
            except NoTranscriptFound:
                return None
        
        # Get the actual transcript
        transcript_data = transcript.fetch()
        subtitles = ' '.join([entry['text'] for entry in transcript_data])
        return subtitles
        
    except (TranscriptsDisabled, VideoUnavailable) as e:
        print(f"Error: {str(e)}")
        return None
    except ValueError as e:
        print(f"Error: {str(e)}")
        return None

def obtain_subtitles():
    video_url = "https://youtu.be/bFJL5LGx2E4"
    # Try with both English and French subtitles
    subtitles = get_youtube_subtitles(video_url, languages=['en', 'fr'])
    if subtitles:
        print("Subtitles:\n", subtitles)
    else:
        print("No subtitles available for this video.")

if __name__ == "__main__":
    main()