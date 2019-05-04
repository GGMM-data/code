import torch

model = torch.nn.Conv2d(1, 6, 5)

input = torch.randn(16, 1, 32, 32)
output = model(input)
print(output.size())
