from __future__ import print_function, division
import torch
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import datasets

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

##########
# Contents
#   Pandas and face dataset, 
#   torch.utils.data.Dataset, __len__, __getitem__ method
#   torchvision.transfroms, class Rescale, RandomCrop, ToTensor 
#   torchvision.transforms.Compose 
#



######################
# pandas and face dataset

landmarks_frame = pd.read_csv('faces/face_landmarks.csv') # pandas frame
print(landmarks_frame)

# id, name, x0, y0, ..., x_67, y_67 

n = 65  # random set n = 65, 
img_name = landmarks_frame.iloc[n, 0] # image name of n = 65
print(img_name)
print(type(img_name))

landmarks = landmarks_frame.iloc[n, 1:].as_matrix() # landmarks points
print(landmarks)
print(type(landmarks))
print(landmarks.shape)

landmarks = landmarks.astype('float').reshape(-1,2)
#print(landmarks)
print(landmarks.shape)


def show_landmarks(image, landmarks):
    """ show image with landmarks
    """
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', color='red')
    #plt.pause(0.001)
    plt.pause(1)

# plt.figure()
# show_landmarks(io.imread(os.path.join('faces',img_name)),landmarks)
# plt.show()


##################
# Dataset class
# torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:
# __len__ so that len(dataset) return the size of the dataset
# __getitem__ to support indexing such that dataset[i] can be used to get sample

class FaceLandmarkDataset(Dataset):
    """Face Landmarks dataset """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string):
            root_dir (string):
            transfrom (callable, optional):
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_name = self.landmarks_frame.iloc[idx, 0]
        image_path = os.path.join(self.root_dir, image_name)
        image = io.imread(image_path)

        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1,2)

        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarkDataset(csv_file="faces/face_landmarks.csv", root_dir="faces")
for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title("Sample #{}".format(i))
    ax.axis("off")
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


#####################
# Transforms is preprocess to get the images of a fixed size
#   Rescale: to scale the image
#   RandomCrop: to crop from image randomly, data augmentation
#   ToTensor: to convect the numpy images to torch images

# We will write them as callable class instead of simple functions so that parameters of the transform need not be passed everytime it's called. Implement __call__ method and if required, __init__ method.


class Rescale(object):
    """Rescale the image in a sample to a given size
    Args:
        output_size (tuple or int)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self,sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w))
        
        # h and w are swapped for landmarks because for images, x and y axes are axis 1 and axis 0 respectively
        landmarks = landmarks * [new_w/w, new_h/h]
        
        return {'image':image, 'landmarks':landmarks}


class RandomCrop(object):
    """ Crop randomly the image in a sample
    Args:
        output_size(tuple or int): Desired output size. If int, squre crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple, int))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top+new_h, left:left+new_w]
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # numpy image: H x W x C
        # torch image: C x H x W
        # convert numpy image to torch image
        image = image.transpose(2,0,1)

        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

#####################
# torchvision.transfroms.Compose

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

# fig = plt.figure()
# sample = face_dataset[65]
# for i,tsfrm in enumerate([scale, crop, composed]):
#     transformed_sample = tsfrm(sample)
#     print(type(transformed_sample))
#     print(type(transformed_sample['image']))
#     print(transformed_sample['image'].size)
#     print(transformed_sample['image'].shape)

#     ax = plt.subplot(1, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     show_landmarks(**transformed_sample)

# plt.show()

####################################
# torchvision.DataLoader
#
# As before, we can use 
#   for data in range(len(FaceLandmarkDataset))
# but we missing out on:
#   Bathching the data
#   Shuffling the data
#   Load the data in parallel using multiprocessing workers

transformed_dataset = FaceLandmarkDataset(csv_file="faces/face_landmarks.csv", root_dir="faces", transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())
    if i == 3:
        break

dataloader = DataLoader(dataset=transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

def show_landmarks_batch(sample_batched):
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)

    image_size = images_batch.size(2)
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * image_size, landmarks_batch[i, :, 1].numpy(), s=1, marker='.', color='r')
        plt.title('Batch from dataloader')



for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())

    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis("off")
        plt.ioff()
        #plt.show()
        break

#########################################
## torchvision

data_transform = transforms.Compose([transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train', transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(dataset=hymenoptera_dataset, batch_size=4, shuffle=True, num_workers=4)

print(type(dataset_loader))

