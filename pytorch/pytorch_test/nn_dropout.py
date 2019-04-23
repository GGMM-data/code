import torch 
import torch.nn as nn

m = nn.Dropout2d(0.3)
inputs=torch.randn(1,28,28)
outputs = m(inputs)
print(outputs)
print(m)

