import torch.nn as nn

def init_weights(m):
  print(m)
  if type(m) == nn.Linear:
    m.weight.data.fill_(1.0)
    print(m.weight)

net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)

for mod in net.children():
   print(mod)

