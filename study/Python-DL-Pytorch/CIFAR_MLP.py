import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'torch.__version__:{torch.__version__}, Device: {DEVICE}')

BATCH_SIZE = 32
EPOCHS = 10

train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                  train = True,
                                  download = True,
                                  transform = transforms.ToTensor())


# train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
#                                             batch_size = BATCH_SIZE,
#                                             shuffle = True)