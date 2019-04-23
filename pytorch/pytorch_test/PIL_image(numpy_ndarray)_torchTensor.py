import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets


dataset = datasets.MNIST('../data', download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4) 

for data in dataloader:
  print(data)
