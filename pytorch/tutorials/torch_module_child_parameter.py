import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
   def __init__(self, input_channel, output_size):
     super(Net, self).__init__()
     self.conv1 = nn.Conv2d(input_channel, 32, 3, padding=1, stride=2)
     self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
     self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
     self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
     self.linear1 = nn.Linear(32 * 3 * 3, 100)
     self.linear2 = nn.Linear(100, 10)

   def forward(self, x):
     x = self.conv1(x)
     x = self.conv2(x)
     x = self.conv3(x)
     x = self.conv4(x)
     x = x.view(-1, 32 * 3 * 3)
     x = F.relu(self.linear1(x))
     x = F.relu(self.linear2(x))
     return x


if __name__ == "__main__":
   batch_size = 32
   input_channel = 1
   output_size = 10
   net = Net(input_channel, output_size)
   inputs = torch.randn(batch_size, input_channel, 42, 42) 
   predict_outputs = net(inputs)

   for module in net.modules():
      print(module)
   for module in net.named_modules():
      print(module)

   for child in net.children():
       print(child)

   for parameter in net.parameters():
       print(parameter.size())
   count = 0
   for parameter in net.named_parameters():
       print(type(parameter))
       print(len(parameter))
       print(parameter[0])
       print(parameter[1].size())
       count+=1
       if count == 2:
           break


