import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# define model

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16, 5, 5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forwrad(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# initialize a model
model = TheModelClass()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# print model's state dict
print("model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# print optimizer's state dict
print("optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# 1. save/load state_dict 
# save model state_dict
PATH = "./saving_model/model.pth"
torch.save(model.state_dict(), PATH)

# load model state_dict
#model = TheModelClass(*args, **kwargs)
model1 = TheModelClass()
model1.load_state_dict(torch.load(PATH))
model1.eval()  # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

# 2. save/load entire model
# save
PATH2 = "./saving_model/model2.pth"
torch.save(model, PATH2)
# load
model2 = TheModelClass()
model2 = torch.load(PATH2)
model2.eval()

# 3. save/load a general checkpoint for inference
# save
PATH3 = "./saving_model/model3.pth"
torch.save({#'epoch':epoch,
	    'model_state_dict':model.state_dict(),
            #'loss': loss,
            'optimizer_state_dict':optimizer.state_dict(),
            }, PATH3)

# load
model3 = TheModelClass()
optimizer3 = optim.SGD(model.parameters(), lr=0.001)
cp = torch.load(PATH3)
model3.load_state_dict(cp['model_state_dict'])
optimizer3.load_state_dict(cp['optimizer_state_dict'])
#epoch = cp['epoch']
#loss = cp['loss']
model3.eval()
# or 
# model3.train()
