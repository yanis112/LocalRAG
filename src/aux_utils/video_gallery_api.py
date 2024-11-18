import requests


def download_videos(keyword: str, n_videos: int, orientation: str = None, size: str = None):
    """
    Downloads videos from the Pexels API based on the given keyword and saves them locally.
    Args:
        keyword (str): The search keyword to find relevant videos.
        n_videos (int): The number of videos to download.
        orientation (str, optional): The orientation of the videos (e.g., 'portrait', 'landscape'). Defaults to None.
        size (str, optional): The size of the videos (e.g., 'small', 'medium', 'large'). Defaults to None.
    Raises:
        Exception: If the request to the Pexels API fails or if video download fails.
    Example:
        download_videos("nature", 5, orientation="landscape", size="medium")
    """
    API_KEY = 'GTpWYa4yf3uIa5mdHrt4vXwf1jwdSvBzWx1b4H2PQQItrm4KtQG7L6Zz'  # Replace with your Pexels API key
    url = f"https://api.pexels.com/videos/search?query={keyword}&per_page={n_videos}"
    
    if orientation:
        url += f"&orientation={orientation}"
    if size:
        url += f"&size={size}"
    
    headers = {
        "Authorization": API_KEY
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['videos']:
            for i, video in enumerate(data['videos']):
                video_url = video['video_files'][0]['link']
                video_response = requests.get(video_url)

                if video_response.status_code == 200:
                    video_filename = f"{keyword.replace(' ', '_')}_{i+1}.mp4"
                    with open(video_filename, 'wb') as video_file:
                        video_file.write(video_response.content)
                    print(f"Video {i+1} downloaded successfully as {video_filename}")
                else:
                    print(f"Failed to download video {i+1}.")
        else:
            print("No videos found for the given keyword.")
    else:
        print("Failed to fetch data from Pexels API.")

if __name__ == "__main__":
    # Example usage
    download_videos("neural networks", 1, orientation="landscape", size="large")