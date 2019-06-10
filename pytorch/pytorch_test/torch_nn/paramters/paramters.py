import torch
import torch.nn as nn


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.cnn1_weight = nn.Parameter(torch.rand(16, 1, 5, 5))
        self.bias1_weight = nn.Parameter(torch.rand(16))
        
        self.cnn2_weight = nn.Parameter(torch.rand(32, 16, 5, 5))
        self.bias2_weight = nn.Parameter(torch.rand(32))
        
        self.linear1_weight = nn.Parameter(torch.rand(4 * 4 * 32, 10))
        self.bias3_weight = nn.Parameter(torch.rand(10))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = F.conv2d(x, self.cnn1_weight, self.bias1_weight)
        out = F.relu(out)
        out = F.max_pool2d(out)
        
        out = F.conv2d(x, self.cnn2_weight, self.bias2_weight)
        out = F.relu(out)
        out = F.max_pool2d(out)
        
        out = F.linear(x, self.linear1_weight, self.bias3_weight)
        return out
