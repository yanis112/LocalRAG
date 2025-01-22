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
from gradio_client import Client, handle_file
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


def process_video(video_path: str, output_folder: str, client: Client, prompt: str = "Hello!!",
                  negative_prompt: str = "music", seed: float = -1, num_steps: int = 25,
                  cfg_strength: float = 4.5, duration: int = 8, add_prefix: str = "sfx_",
                  auto_prompt: bool = False) -> str:
    """Processes a single video using the MMAudio Gradio API.

    Args:
        video_path: The path to the input video file.
        output_folder: The path to the output folder where the processed video will be saved.
        client: The Gradio client instance.
        prompt: The text prompt for audio generation.
        negative_prompt: The negative prompt for audio generation.
        seed: The seed for random number generation (-1 for random).
        num_steps: The number of steps for audio generation.
        cfg_strength: The guidance strength for audio generation.
        duration: The duration of the audio in seconds.
        add_prefix: Prefix to add to the output video file name.
        auto_prompt: If True, generate a prompt automatically using image analysis.

    Returns:
        The path to the processed video file if successful.

    Raises:
        RuntimeError: If the Gradio API call fails or the result is unexpected.
    """
    try:
        if auto_prompt:
            try:
              image_analyzer = ImageAnalyzerAgent()
              #Ensure that the output folder exist before creating the frame
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



        result = client.predict(
            video={"video": handle_file(video_path)},
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_steps=num_steps,
            cfg_strength=cfg_strength,
            duration=duration,
            api_name="/predict"
        )

        if not isinstance(result, dict) or "video" not in result:
            raise RuntimeError(f"Unexpected result format from Gradio API: {result}")

        output_video_path = result["video"]

        if not output_video_path:
            raise RuntimeError(f"No output video returned from the Gradio API:{result}")

        # Extract file name
        video_file_name = os.path.basename(video_path)
        video_file_name_without_ext, ext = os.path.splitext(video_file_name)

        # Create output folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)

        # Construct the output file path
        final_output_path = os.path.join(output_folder, f"{add_prefix}{video_file_name_without_ext}{ext}")

        # Check if a file is returned or a path
        if os.path.isfile(output_video_path):
            os.rename(output_video_path, final_output_path)
        elif output_video_path.startswith("/"):
            # Copy the file in case it is an absolute path
            import shutil
            shutil.copy2(output_video_path, final_output_path)
        else:
            raise RuntimeError(f"the returned file path is invalid : {output_video_path}")

        # Remove temporary first_frame file
        if os.path.exists(first_frame_path):
            os.remove(first_frame_path)
        return final_output_path

    except Exception as e:
        raise RuntimeError(f"Error processing video: {e}")


def process_videos(input_path: str, output_folder: str, prompt: str = "Hello!!",
                   negative_prompt: str = "music", seed: float = -1, num_steps: int = 25,
                   cfg_strength: float = 4.5, duration: int = 8, add_prefix: str = "sfx_",
                   auto_prompt: bool = False):
    """Processes a video file or all videos in a folder using the MMAudio Gradio API.

    Args:
        input_path: The path to the input video file or folder.
        output_folder: The path to the output folder where the processed videos will be saved.
        prompt: The text prompt for audio generation.
        negative_prompt: The negative prompt for audio generation.
        seed: The seed for random number generation (-1 for random).
        num_steps: The number of steps for audio generation.
        cfg_strength: The guidance strength for audio generation.
        duration: The duration of the audio in seconds.
        add_prefix: Prefix to add to the output video file names.
        auto_prompt: If True, generate a prompt automatically using image analysis.
    """
    client = Client("hkchengrex/MMAudio")

    if os.path.isfile(input_path):
        try:
            output_path = process_video(input_path, output_folder, client, prompt, negative_prompt, seed, num_steps,
                                      cfg_strength, duration, add_prefix, auto_prompt)
            print(f"Processed video saved at: {output_path}")
        except RuntimeError as e:
            print(f"Error processing video {input_path} : {e}")

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
        print(f"Invalid input path: {input_path}. Please provide a valid file or folder path.")


if __name__ == "__main__":
    # Example Usage
    input_path = r"C:\Users\Yanis\Documents\AI Art\Doom Of Valyria\videos\kling-1.6"
    #"aux_data/test_2.mp4"  # Replace with your video file path or folder path
    # input_path = "test_folder"  # Example folder
    output_directory = r"C:\Users\Yanis\Documents\AI Art\Doom Of Valyria\videos\sfx_videos"  # Replace with your output folder path
    add_prefix_str = "sfx_"
    auto_generate_prompt = True # Set to True to use image analysis for prompts


    process_videos(input_path, output_directory, prompt="A funny narration", duration=5, add_prefix=add_prefix_str,
                 auto_prompt=auto_generate_prompt)
    # Clean up dummy test video if created
    if not os.path.exists(input_path) and os.path.exists("test_video.mp4"):
        os.remove("test_video.mp4")