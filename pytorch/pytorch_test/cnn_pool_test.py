import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # define cnn layers
        self.conv1 = nn.Conv2d(1, 6, 5) # input_channels, output_channels, filter_size
        self.conv2 = nn.Conv2d(6, 16, 5)

        # define affine layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        # input x is image, have shape (,)
        print(x.size())
        # first cnn, max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print(x.size())

        # second cnn, max pooling, if the pooling size is a square, specify just a single number, here is 2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(x.size()) 

        # flatten
        x = x.view(-1, self.num_flat_features(x))

        # first fully-connect layer
        x = F.relu(self.fc1(x))
         
        # second fully-connect layer
        x = F.relu(self.fc2(x))

        # output layer
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())


input = torch.randn(1, 1, 32, 32)
output = net(input)
print(output)
