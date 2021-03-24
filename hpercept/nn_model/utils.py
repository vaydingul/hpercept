import torch

def get_device():
    """

    If a GPU is available, then it set "device" to "cuda:0"
    else, it set "device" to "cpu".

    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device