import torch

def get_torch_device():
    # Prefer MPS (Apple Silicon) when available, then CUDA, otherwise CPU.
    # torch.backends.mps.* checks ensure PyTorch was built with MPS support.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    return device

