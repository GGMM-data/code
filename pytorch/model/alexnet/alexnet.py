import torch.nn as nn
import torch
import torch.nn.functional as F

class AlexNet(nn.Module):
     def __init__(self, output_size=1000):
         super(AlexNet, self).__init__()
         self.conv1 = nn.Conv2d(3, 96, 11, 4)
         self.pool1 = nn.MaxPool2d(3, 2)
         self.conv2 = nn.Conv2d(96, 256, 5)
         self.pool2 = nn.MaxPool2d(3, 2)
         self.conv3 = nn.Conv2d(256, 384, 3)
         self.conv4 = nn.Conv2d(384, 384, 3)
         self.conv5 = nn.Conv2d(384, 256, 3)
         self.pool3 = nn.MaxPool2d(3, 2)
         self.drop1 = nn.Dropout()
         self.fc1 = nn.Linear(256 * 1 * 1, 4096) 
         self.drop2 = nn.Dropout()
         self.fc2 = nn.Linear(4096, 4096)
         self.fc3 = nn.Linear(4096, output_size)

     def forward(self, x):
         x = F.relu(self.conv1(x))
         x = self.pool1(x)
         x = F.relu(self.conv2(x))
         x = self.pool2(x)
         x = F.relu(self.conv3(x))
         x = F.relu(self.conv4(x))
         x = F.relu(self.conv5(x))
         x = self.pool3(x)
         x = x.view(-1, 256 * 1 * 1)
         x = self.drop1(x)
         x = F.relu(self.fc1(x))
         x = self.drop2(x)
         x = F.relu(self.fc2(x))
         y = self.fc3(x)

         return y


if __name__ == "__main__":
    x = torch.randn([16, 3, 224, 224])
    alex = AlexNet()
    print(alex)
    y = alex(x)
    print(y.size())

