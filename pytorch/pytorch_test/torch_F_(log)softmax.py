import torch
import torch.nn.functional as F


data = torch.Tensor([1,2,3])
print(data)
softmax = F.softmax(data, dim=0)
print(softmax)
print(sum(softmax))
log_softmax = F.log_softmax(data, dim=0)
print(log_softmax)
print(sum(log_softmax))