import torch
from torch import nn
from torch.autograd import Variable

input = Variable(torch.randn(30,20,32,32))
print(input.size())
# output: torch.Size([30, 20, 32, 32])

m1 = nn.MaxPool2d(2)
output = m1(input)
print(output.size())
# torch.Size([30, 20, 16, 16])

m2 = nn.MaxPool2d(5)
print(m2)
# output: MaxPool2d (size=(5, 5), stride=(5, 5), dilation=(1, 1))

for param in m2.parameters():
  print(param)

print(m2.state_dict().keys())
# output: []

output = m2(input)
print(output.size())
# output: torch.Size([30, 20, 6, 6])

