import torch
import torch.nn as nn


model = nn.Sequential(nn.Linear(5, 3), nn.Sequential(nn.Linear(3, 2)))

for module in model.modules():
    print(module)


print("===========================")
for child in model.children():
    print(child)
