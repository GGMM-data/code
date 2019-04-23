import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from myown.torch.model.alexnet_change import AlexNet
import argparse
import matplotlib.pyplot as plt

# 因为
def parse_args():
    parser = argparse.ArgumentParser("input parameters")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    args_list = parser.parse_args()
    return args_list

def train(args, model):
    transform1 = torchvision.transforms.ToTensor()
    transforms = torchvision.transforms.Compose([transform1])
    trainset = torchvision.datasets.CIFAR100(root="./datasets/", train=True, transform=transforms, download=False)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args_list.batch_size, shuffle=True,
                                               num_workers=4)
    for episode in range(args.episodes):
        plt.figure()
        plt.ion()
        step_list = []
        loss_list = []
        for i, data in enumerate(train_loader):
            imgs, labels = data
            labels_predicted = model(imgs)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=args_list.lr, momentum=args.momentum)

            loss = criterion(labels_predicted, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("step ", i, " : ", loss.item())
                step_list.append(i)
                loss_list.append(loss.item())
                plt.plot(step_list, loss_list, linewidth=1)
                # plt.pause(0.1)
        plt.ioff()
        plt.show()

        plt.savefig("./alexnet_change_test/episode_"+str(episode)+"_loss.png")


def test(model):
    transform1 = torchvision.transforms.ToTensor()
    transforms = torchvision.transforms.Compose([transform1])
    testset = torchvision.datasets.CIFAR100(root="./datasets/", train=False, transform=transforms, download=False)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=args_list.batch_size, shuffle=False,
                                              num_workers=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            imgs, labels = data
            label_predicted = model(imgs)
            _, predicted = torch.max(label_predicted, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # .item() to get the value of one element tensor
    print("Accuracy is %d %%" % (correct * 100 / total))


def main(args_list):
    alexnet_change = AlexNet(output_size=100)
    train(args_list, alexnet_change)
    test(alexnet_change)



if __name__ == "__main__":
    args_list = parse_args()
    main(args_list)
