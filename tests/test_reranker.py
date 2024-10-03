import subprocess
import threading
import time
from src.retrieval_utils import load_reranker

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

def test_reranker_model():
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_vram_usage, args=(stop_event,))
    
    print("Starting VRAM monitoring...")
    monitor_thread.start()

    print("Loading reranker model...")
    reranker = load_reranker("jinaai/jina-reranker-v2-base-multilingual")
    assert reranker is not None
    
    print("Making inference...")
    query = "What is the capital of France?"
    docs = [(query, "Paris is the capital of France"), (query, "The capital of France is Paris")]
    scores = reranker.score(docs)
    
    print("Stopping VRAM monitoring...")
    stop_event.set()
    monitor_thread.join()

    print("Scores:", scores)
    return scores

if __name__ == "__main__":
    test_reranker_model()