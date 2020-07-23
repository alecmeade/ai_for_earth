import torch
import gc


def clear_cuda():
    """Clears the gpu cache and triggers garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()

def maybe_get_cuda_device() -> torch.device:
    """Retrieves a cuda GPU device if available otherwise a cpu device.

    Returns:
        A torch.device corresponding to a gpu or cpu.
    """
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  

    return torch.device(dev)  
