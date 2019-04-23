import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # define cnn layers
        # self.conv0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=0) 
        self.conv1 = nn.Conv2d(1, 6, 5) # input_channels, output_channels, filter_size
        self.conv2 = nn.Conv2d(6, 16, 5)
# define affine layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        # input x is image, have shape (,)
        print(x.size())
        
        # x = self.conv0(x)
        # print(x.size())

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
#print(params)

input = torch.randn(1, 1, 32, 32)
output = net(input)
print(output)

############################################
# loss function

target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# input -> Conv2d -> relu -> max_pooling -> Conv2d -> relu -> max_pooling 
#   -> view -> linear -> relu -> linear -> relu -> linear
#   -> MSELoss
#   -> Loss

print("MSELoss")
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[1][0]) # ReLU
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0]) # ReLU
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0]) # View
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0]) # MaxPool2d
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[0][0]) # ReLU
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[0][0].next_functions[0][0]) # CNN
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0]) # MaxPool2d
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0]) # ReLU
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0]) # CNN
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0]) # None
#....
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0]) # tuple index out of range 

##################################################
## backward

net.zero_grad() # zero the gradient buffer of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
print(net.conv1.weight.grad)
print(net.conv2.bias.grad)
print(net.conv2.weight.grad)
print(net.fc1.bias.grad)
print(net.fc1.weight.grad)
print(net.fc2.bias.grad)
print(net.fc2.weight.grad)
print(net.fc3.bias.grad)
print(net.fc3.weight.grad)

loss.backward()

print('conv1.bias.grad after backward')

print(net.conv1.bias.grad.size())
print(net.conv1.weight.grad.size())
print(net.conv2.bias.grad.size())
print(net.conv2.weight.grad.size())
print(net.fc1.bias.grad.size())
print(net.fc1.weight.grad.size())
print(net.fc2.bias.grad.size())
print(net.fc2.weight.grad.size())
print(net.fc3.bias.grad.size())
print(net.fc3.weight.grad.size())

#net.zero_grad() #zero the gradient buffers of all parameters
#output.backward(torch.randn(1, 10)) # backprops with random gradients

#output.backward() # error! since output is a tensor, not a scalar, it must specify the tensor

#####################
# update the weigths
# SGD
# net.named_parameters() = (name, net.parameters())

# update the parameters by hands
learning_rate = 0.01
print(type(net.parameters()))
f = list(net.parameters())[0]
print(type(f))

for f in net.named_parameters():
    print(f[0])
    f[1].data.sub_(f[1].grad.data * learning_rate)

# update the parameters by the optimizer

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=learning_rate)
# loss2 = criterion(output, target)
# loss2.backward()
# optimizer.step() # doing the update
# since the gradient has been calculated
