import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


# tensor = torch.randn(3, 4)
# device = torch.device("cuda:0")
# model.to(device) # put a model oh the gpu
# model = nn.DataParallel(model) # pytorch will only use one gpu defaultly, use DataParallel to run operations on multiple gpus
# tensor_gpu = tensor.to(device)

# parameters and dataloaders

input_size = 5
output_size = 2

batch_size = 30
data_size = 100


# create a dummy dataset
class RandomDataset(Dataset):
    def __init__(self, length, size):
        self.len = length
        self.data = torch.randn(length, size)
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(data_size, input_size), batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        print("\tIn Model: input size", inputs.size(), "output size", outputs.size())
        return outputs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model) #model, device_ids=[0,1,2]
model.to(device)

for data in rand_loader:
    inputs = data.to(device)
    outputs = model(inputs)
    print("Outside: input size", inputs.size(), "output size", outputs.size())
