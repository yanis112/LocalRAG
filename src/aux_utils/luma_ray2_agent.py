import os
import time
import requests
from lumaai import LumaAI
from urllib.parse import urlparse
from os.path import splitext, basename

class Image2VideoAgent:
    def __init__(self):
        self.client = LumaAI(auth_token=os.getenv("LUMA_API_KEY"))

    def generate_video(self,prompt, image_path, output_folder, resolution="1080p", duration="5s"):
        """
        Generates a video from a reference image using the Luma AI API.

        Args:
            image_path (str): Path to the reference image.
            output_folder (str): Path to the folder where the video will be saved.
            resolution (str): Resolution of the video (e.g., "720p").
            duration (str): Duration of the video (e.g., "5s").
        """
        try:
            # Upload image to a CDN and get the URL
            image_url = self._upload_image_to_cdn(image_path)  # Implement this

            # Extract filename from path
            filename = basename(image_path)
            name, ext = splitext(filename)
            output_filename = f"ray2_{resolution}_{name}.mp4"
            output_path = os.path.join(output_folder, output_filename)

            generation = self.client.generations.create(
                prompt=prompt,
                model="ray-2",
                resolution=resolution,
                duration=duration,
                keyframes={
                    "frame0": {
                        "type": "image",
                        "url": image_url
                    }
                }
            )
            print(f"API Request Payload: {generation}") # Debugging line

            completed = False

            completed = False
            while not completed:
                generation = self.client.generations.get(id=generation.id)
                if generation.state == "completed":
                    completed = True
                elif generation.state == "failed":
                    raise RuntimeError(f"Generation failed: {generation.failure_reason}")
                print("Dreaming")
                time.sleep(3)

            video_url = generation.assets.video

            # Download the video
            response = requests.get(video_url, stream=True)
            with open(output_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded as {output_path}")

        except Exception as e:
            print(f"Error generating video: {e}")

    def _upload_image_to_cdn(self, image_path):
        """
        Placeholder for image upload to CDN.  You'll need to implement this.
        For example, you could use AWS S3, Google Cloud Storage, or a similar service.
        For local testing, you could use a service like ngrok to expose a local server.
        """
        # TODO: Implement image upload to a CDN and return the URL
        # This is a placeholder - replace with your actual implementation
        print("Warning: No CDN upload implemented.  Using a placeholder URL.")
        print("Make sure this URL is a valid CDN URL for Luma AI to process.") # Debugging line
        return "https://example.com/placeholder_image.jpg"

if __name__ == "__main__":
    import time
    start_time = time.time()
    # Replace with your actual API key and image path
    image_path = "aux_data/test_image.jpg"  # Replace with your image path
    output_folder = "output_vid"  # Replace with your desired output folder
    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    agent = Image2VideoAgent()
    agent.generate_video(prompt="Blockbuster movie scene", image_path=image_path, output_folder=output_folder)
    print(f"Total time: {time.time() - start_time} seconds")
