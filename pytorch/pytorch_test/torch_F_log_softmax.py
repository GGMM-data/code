import torch
import torch.nn.functional as F


data = torch.Tensor([1, 1, 0, 0, 0])
print(data)

# 1.softmax
# 计算公式 $\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}$
softmax = F.softmax(data, dim=0)
print(softmax)
print(sum(softmax))

# 2.log_softmax
# 计算公式 log_softmax = log(softmax)
log_softmax = F.log_softmax(data, dim=0)
print(log_softmax)
print(sum(log_softmax))