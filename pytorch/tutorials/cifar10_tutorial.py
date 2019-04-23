import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

##################
# 1.load data

# output of torchvision datasets are PILImages of range[0,1]
# we transform Tensors of normailzed range[-1,1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) # torchvision.transforms.Compose(), compose serveral transforms together
# transforms.Normalize((mean1,mean2,...),(std1,std2,...))

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size(),labels.size())

imshow(torchvision.utils.make_grid(images))
#plt.show()
print(' '.join("%5s" % classes[labels[j]] for j in range(4)))
print(type(images))
print(images.size())

dataiter = iter(testloader)
images, labels = dataiter.next()
print(images.size(),labels.size())


#############################
print("define network")

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
for i,f in enumerate(net.named_parameters()):
    print("parameter %d" % (i+1))
    print(f[0])
    print(f[1].data.size())

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#
#         outputs = net(images)
#         _,predicted =  torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print("Accuracy is %d %%" % (correct*100/total))

device = torch.device("cuda:0" if torch.cuda.is_available() else "gpu")
print(device)
net.to(device)

begin_time = time.time()
for epoch in range(2):
    
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        optimizer.zero_grad() # zero gradients

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs) # predict outpus
        
        loss = criterion(outputs, labels)
        loss.backward() # calculate gradients
        #print("batch %5d loss is %.3f" % (i + 1, loss))

        optimizer.step() # update the weights
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print("Every 2000 batches, [%d,%5d] loss %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0

end_time = time.time()
total_time = end_time - begin_time
print("Total train time is %.2f" % (total_time))

correct = 0
total = 0
begin_time = time.time()

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        _,predicted =  torch.max(outputs, 1)
        total += labels.size(0)
        # temp = (predicted == labels) # it's a tensor 
        correct += (predicted == labels).sum().item() # .item() to get the value of one element tensor
print("Accuracy is %d %%" % (correct*100/total))

end_time = time.time()
total_time = end_time - begin_time
print("Total test time %.2f" % (total_time) )

class_correct = [0. for i in range(10)]
total_correct = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1

for i in range(10):
    print("Accuracy of %5s : %2d %%" % (classes[i], 100 * class_correct[i]/total_correct[i]))
