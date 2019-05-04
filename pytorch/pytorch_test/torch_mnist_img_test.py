from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="MNIST data")
parser.add_argument("--batch_size", type=int, default=2, metavar='n',
                    help='input batch size for tranining (default: 2)')
args = parser.parse_args()
train_loader = torch.utils.data.DataLoader(
                   datasets.MNIST("../data", download=False,
                         transform=transforms.Compose(
                             [transforms.ToTensor(),
                              transforms.Normalize((0.1307,),(0.3081,)),
                             ])),
                   batch_size=args.batch_size,
                   shuffle=True,
                   )
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.size(), labels.size())
image_sample = images[0]
print(image_sample.max())
print(image_sample.min())
#print(image_sample)
for i in range(args.batch_size):
  plt.figure()
  image = images[i]
  label = labels[i]
  image = torch.squeeze(image, 0)
  numpy_img = image.numpy()
  plt.imshow(numpy_img)

plt.show()


train_loader = torch.utils.data.DataLoader(
                   datasets.MNIST("../data", download=False,
                         transform=transforms.Compose(
                             [transforms.ToTensor(),
                              #transforms.Normalize((0.1307,),(0.3081,)),
                             ])),
                   batch_size=args.batch_size,
                   shuffle=True,
                  )
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.size(), labels.size())
image_sample = images[0]
print(image_sample.max())
print(image_sample.min())
#print(image_sample)

for i in range(args.batch_size):
  plt.figure()
  image = images[i]
  label = labels[i]
  image = torch.squeeze(image, 0)
  numpy_img = image.numpy()
  plt.imshow(numpy_img)

plt.show()
