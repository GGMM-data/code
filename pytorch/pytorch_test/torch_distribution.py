import torch.distributions as diss
import torch

# 手动设置torch随机数种子，让结果可以复现
torch.manual_seed(5)

# Categorical
m = diss.Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25 ]))
for _ in range(5):
    print(m.sample())

m = diss.Categorical(torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]))
for _ in range(5):
    print(m.sample())

