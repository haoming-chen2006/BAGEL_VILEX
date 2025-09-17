import torch
print(f"Number of GPUs available: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")