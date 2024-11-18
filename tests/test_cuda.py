import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}" if torch.cuda.is_available() else "Using CPU")