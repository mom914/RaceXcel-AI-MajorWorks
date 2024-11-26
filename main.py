import pygame
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


if torch.cuda.is_available:
    device = "cuda"
elif torch.mps.is_available:
    device = "mps"
else:
    device = "cpu"
print(f" using: {device}")
