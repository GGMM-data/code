import torch 
import torch.nn.functional as F

x = torch.randn(1, 28, 28)
print(type(x))
y = F.dropout(x, 0.5, True)
#y = F.dropout2d(x, 0.5)

print(y)
