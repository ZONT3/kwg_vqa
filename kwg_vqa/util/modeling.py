import torch
from torch import nn


class Module(nn.Module):
    """Необходимая база для всех модулей проекта"""

    def __init__(self, args):
        self.args = args
        self.device = detect_cuda_device(args)


def detect_cuda_device(args):
    if not args.device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return args.device
