import torch
import torch.nn as nn
from torch.autograd import Variable

inputs = Variable(torch.randn(64, 3, 32, 32))

m1 = nn.Conv2d(3, 16, 3)
print(m1)
# output: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
output1 = m1(inputs)
print(output1.size())
# output: torch.Size([64, 16, 30, 30])

m2 = nn.Conv2d(3, 16, 3, padding=1)
print(m2)
# output: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
output2 = m2(inputs)
print(output2.size())
# output: torch.Size([64, 16, 32, 32])
