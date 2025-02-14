"""
This script uses the Gradio API to process video(s), adding a generated audio track based on text prompts.

It takes a path to a video file or a folder containing videos as input, and an output directory.
It then uses the MMAudio Gradio space to generate audio and combine it with the input video(s).
The resulting video(s) are saved to the specified output directory with an added prefix.
A prompt can be given or auto-generated using an IA image analyzer.
"""
import os
import glob
import time
from tqdm import tqdm
from gradio_client import Client
from moviepy import VideoFileClip
from PIL import Image
from io import BytesIO

# Import the ImageAnalyzerAgent from the specified path
from src.aux_utils.vision_utils_v2 import ImageAnalyzerAgent




def extract_first_frame(video_path: str, output_path: str):
    """Extracts the first frame of a video and saves it as an image.

    Args:
        video_path: The path to the input video file.
        output_path: The path to save the output image file.

    Returns:
        str: The output_path where the image has been saved.

    Raises:
         RuntimeError: If the video loading fails or the frame extraction fails.
    """
    try:
        clip = VideoFileClip(video_path)
        if clip.duration <= 0:
            raise RuntimeError(f"Invalid video duration for : {video_path}")

        frame = clip.get_frame(0)
        image = Image.fromarray(frame)
        image.save(output_path)
        clip.close()
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error extracting first frame: {e}")

# Updated the function to use the proper gradio_client API
def process_video(video_path: str, output_folder: str, client: Client, prompt: str = "Hello!!",
                  negative_prompt: str = "music", seed: float = -1, num_steps: int = 25,
                  cfg_strength: float = 4.5, duration: int = 8, add_prefix: str = "sfx_",
                  auto_prompt: bool = False) -> str:
    """Processes a single video using the MMAudio Gradio API."""
    try:
        if auto_prompt:
            try:
                image_analyzer = ImageAnalyzerAgent()
                os.makedirs(output_folder, exist_ok=True)
                first_frame_path = os.path.join(output_folder, "first_frame.png")
                try:
                    extract_first_frame(video_path, first_frame_path)
                    vision_prompt = "Provide a description (just a few words) of a realistic, cinematic sound effect (SFX) — no music — that would be appropriate for accompanying the provided scene/image, e.g: a rock falling from a cliff, dragon roaring, ect.., answer the sfx description without preamble."
                    prompt = image_analyzer.describe(
                        first_frame_path, prompt=vision_prompt, vllm_provider="gemini",
                        vllm_name="gemini-2.0-flash-exp"
                    )
                    print("######################")
                    print("SOUND EFFECT DESCRIPTION: ", prompt)
                    print("######################")
                except RuntimeError as e:
                    print(f"Error generating prompt, using default : {e}")
            except Exception as e:
                print(f"Error during Image Analyzer initialization, using default prompt : {e}")

        # Check if video file exists
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video file not found: {video_path}")

        # Ensure the video file can be opened
        with VideoFileClip(video_path) as clip:
            if clip.duration <= 0:
                raise RuntimeError(f"Invalid video duration for : {video_path}")

        result = client.predict(
            video_path,  # Now passing the path directly
            prompt,
            negative_prompt,
            seed,
            num_steps,
            cfg_strength,
            duration,
            api_name="/predict"
        )

        if not result or not isinstance(result, tuple) or len(result) == 0:
            raise RuntimeError(f"Unexpected result format from Gradio API: {result}")

        # The API returns the output video path directly
        output_video_path = result[0]

        if not output_video_path or not os.path.exists(output_video_path):
            raise RuntimeError(f"No output video returned from the Gradio API: {result}")

        # Extract file name
        video_file_name = os.path.basename(video_path)
        video_file_name_without_ext, ext = os.path.splitext(video_file_name)

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Construct the output file path
        final_output_path = os.path.join(output_folder, f"{add_prefix}{video_file_name_without_ext}{ext}")

        # Move or copy the output file to final destination
        if os.path.isfile(output_video_path):
            os.rename(output_video_path, final_output_path)
        elif output_video_path.startswith("/"):
            import shutil
            shutil.copy2(output_video_path, final_output_path)
        else:
            raise RuntimeError(f"The returned file path is invalid : {output_video_path}")

        # Remove temporary first_frame file if it exists
        if os.path.exists(first_frame_path):
            os.remove(first_frame_path)

        return final_output_path

    except Exception as e:
        raise RuntimeError(f"Error processing video: {e}")


def process_videos(input_path: str, output_folder: str, prompt: str = "Hello!!",
                   negative_prompt: str = "music", seed: float = -1, num_steps: int = 25,
                   cfg_strength: float = 4.5, duration: int = 8, add_prefix: str = "sfx_",
                   auto_prompt: bool = False):
    """Processes a video file or all videos in a folder using the MMAudio Gradio API."""
    try:
        client = Client("hkchengrex/MMAudio")
        
        if os.path.isfile(input_path):
            output_path = process_video(input_path, output_folder, client, prompt, negative_prompt, seed, num_steps,
                                      cfg_strength, duration, add_prefix, auto_prompt)
            print(f"Processed video saved at: {output_path}")

        elif os.path.isdir(input_path):
            video_files = glob.glob(os.path.join(input_path, "*.mp4"))  #  only look for mp4 files you can add other video extension here
            if not video_files:
                print(f"No mp4 video files found in folder: {input_path}")
                return

            for video_file in tqdm(video_files, desc="Processing Videos"):
                try:
                    output_path = process_video(video_file, output_folder, client, prompt, negative_prompt, seed,
                                           num_steps, cfg_strength, duration, add_prefix, auto_prompt)
                    print(f"Processed video saved at: {output_path}")
                except RuntimeError as e:
                    print(f"Error processing video {video_file} : {e}")
        else:
            raise RuntimeError(f"Invalid input path: {input_path}. Please provide a valid file or folder path.")
            
    except Exception as e:
        raise RuntimeError(f"Error in process_videos: {e}")

if __name__ == "__main__":
    # Example Usage
    input_path = r"C:\Users\Yanis\Documents\AI Art\Doom Of Valyria\videos\kling-1.6"
    output_directory = r"C:\Users\Yanis\Documents\AI Art\Doom Of Valyria\videos\sfx_videos" 
    add_prefix_str = "sfx_"
    auto_generate_prompt = True

    process_videos(input_path, output_directory, prompt="A funny narration", duration=5, add_prefix=add_prefix_str,
                 auto_prompt=auto_generate_prompt)