import torch
import torch.nn as nn


class Model(nn.Module):
   def __init__(self, num_inputs):
      super(Model, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2)
      self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
      self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
      self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)


   def forward(self, x):
      print(x.size())
      x = self.conv1(x)
      print(x.size())
      x = self.conv2(x)
      print(x.size())
      x = self.conv3(x)
      print(x.size())
      x = self.conv4(x)
      print(x.size())


if __name__ == "__main__":
   inputs = torch.randn(1, 1, 42, 42)
   net = Model(1)
   outputs = net(inputs)
     
