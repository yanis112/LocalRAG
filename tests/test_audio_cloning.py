import os
import locale
import subprocess
from huggingface_hub import snapshot_download

# Set locale for Linux users
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Define paths and parameters
root_dir = "/home/user/llms"  # Root directory
fish_speech_dir = os.path.join(root_dir, "fish-speech")
model_dir = os.path.join(fish_speech_dir, "checkpoints/fish-speech-1.4")
output_folder = os.path.join(fish_speech_dir, "output_audio")
os.makedirs(output_folder, exist_ok=True)

# Download model if not exists
snapshot_download(repo_id="fishaudio/fish-speech-1.4", local_dir=model_dir)
print("All checkpoints downloaded")

def encode_reference_audio(reference_audio):
    try:
        result = subprocess.run([
            "python", os.path.join(fish_speech_dir, "tools/vqgan/inference.py"),
            "-i", reference_audio,
            "--checkpoint-path", os.path.join(model_dir, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
            "--output-path", os.path.join(root_dir, "fake.npy")  # Ensure the output path is in the root directory
        ], check=True, capture_output=True, text=True)
        print("Reference audio encoded successfully")
    except subprocess.CalledProcessError as e:
        print("Error encoding reference audio:", e.stderr)

def generate_semantic_tokens(text, prompt_text, prompt_tokens):
    prompt_tokens_path = os.path.join(root_dir, prompt_tokens)  # Ensure the path is correct
    if not os.path.exists(prompt_tokens_path):
        print(f"Error: Path '{prompt_tokens_path}' does not exist.")
        return
    
    result = subprocess.run([
        "python", os.path.join(fish_speech_dir, "tools/llama/generate.py"),
        "--text", text,
        "--prompt-text", prompt_text,
        "--prompt-tokens", prompt_tokens_path,
        "--checkpoint-path", model_dir,
        "--num-samples", "2"
    ], cwd=fish_speech_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Error generating semantic tokens: {result.stderr}")
    else:
        print(result.stdout)

def generate_cloned_audio():
    codes_path = os.path.join(root_dir, "codes_0.npy")  # Ensure the path is correct
    if not os.path.exists(codes_path):
        print(f"Error: Path '{codes_path}' does not exist.")
        return
    
    result = subprocess.run([
        "python", os.path.join(fish_speech_dir, "tools/vqgan/inference.py"),
        "-i", codes_path,
        "--checkpoint-path", os.path.join(model_dir, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
        "--output-path", os.path.join(output_folder, "cloned_audio.wav")
    ], cwd=fish_speech_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Error generating cloned audio: {result.stderr}")
    else:
        print(f"Cloned audio saved in {os.path.join(output_folder, 'cloned_audio.wav')}")

# Example usage
encode_reference_audio("test.wav")
generate_semantic_tokens("hello world", "The text corresponding to reference audio", "fake.npy")
generate_cloned_audio()