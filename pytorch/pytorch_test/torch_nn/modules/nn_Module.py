import torch
import torch.nn as nn

l = nn.Linear(2, 2)
print(type(l.parameters()))
