import subprocess
import threading
import time
from src.main_utils.embedding_utils import get_embedding_model

def get_vram_usage():
    """
    Return the VRAM used at the moment.
    """
    vram_usage = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        .decode()
        .strip()
    )
    vram_usage_gb = float(vram_usage) / 1024  # Convert to GB assuming the output is in MB
    return vram_usage_gb

def monitor_vram_usage(stop_event):
    while not stop_event.is_set():
        vram_usage_gb = get_vram_usage()
        print(f"VRAM usage: {vram_usage_gb:.2f} GB")
        time.sleep(0.1)

def test_get_embedding_model():
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_vram_usage, args=(stop_event,))
    
    print("Starting VRAM monitoring...")
    monitor_thread.start()

    print("Loading model...")
    model = get_embedding_model("BAAI/bge-m3")
    
    print("Making inference...")
    embedding = model.embed_documents("Ceci est un document de test")
    
    print("Stopping VRAM monitoring...")
    stop_event.set()
    monitor_thread.join()

    print("Model:", model)
    #print("Embedding:", embedding)
    print("Length of embedding:", len(embedding))

if __name__ == "__main__":
    test_get_embedding_model()