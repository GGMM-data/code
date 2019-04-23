import torchvision
import torch

# 1.查看torchvision提供的所有datasets
print(torchvision.datasets.__all__)
# ('LSUN', 'LSUNClass', 'ImageFolder', 'DatasetFolder', 'FakeData', 'CocoCaptions', 'CocoDetection', 'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'MNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION', 'Omniglot')

#
#transform1 = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform2 = torchvision.transforms.ToTensor()
transforms = torchvision.transforms.Compose([
                                            # transform1,
                                             transform2 ])

trainset = torchvision.datasets.CIFAR100(root="./datasets/cifar", train=True, transform=transforms, download=False)
testset = torchvision.datasets.CIFAR100(root="./datasets/cifar", train=False, transform=transforms, download=False)

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=16, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=False, num_workers=8)

def train():
    episodes = 1
    print(len(train_loader))
    print(type(train_loader))
    for episode in range(episodes):
        for i, data in enumerate(train_loader):
           images, labels = data
           print(images.size(), labels.size())
           if i > 1:
              break


if __name__ == "__main__":
    train()
    #test()
