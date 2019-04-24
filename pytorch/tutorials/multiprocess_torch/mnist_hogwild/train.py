from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os


def train(rank, args, model, device, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)
    train_loader = DataLoader(datasets.MNIST("../../datasets/mnist/",
                                             download=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]),
                                             ),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=1,
                              **dataloader_kwargs
                            )
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for i, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        predicted_y = model(x)
        loss = F.nll_loss(predicted_y, y.to(device))
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, i * len(x), len(data_loader.dataset),
                            100. * i / len(data_loader), loss.item()))


def test(args, model, device, dataloader_kwargs):
    torch.manual_seed(args.seed)
    test_loader = DataLoader(datasets.MNIST("../../datasets/mnist/",
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]),
                                            train=False,
                                             download=False),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=1,
                              **dataloader_kwargs
                            )
    for epoch in range(1, args.epochs + 1):
        test_epoch(model, device, test_loader)


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            predicted_y = model(x)
            test_loss += F.nll_loss(predicted_y, y.to(device), reduction='sum').item()
            print(predicted_y.size())
            predicted = predicted_y.max(1)[1]
            correct += (predicted.eq(y.to(device)).sum().item())

        test_loss = test_loss / len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
