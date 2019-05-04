import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

#img = np.random.randint(low=0, high=255, size=(3,32,32))
#img = np.random.randint(low=0, high=255, size=(32,32,3))
#img = np.array(img, dtype=np.float32)

transform1 = transforms.ToTensor()
# shape:(*, *, 3)
transform2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# (mean1, mean2, mean3), (std1, std2, std3) 
# normalize: (raw - mean)/std
# 
# official documents: Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input torch.*Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]

img = np.ones((32, 32, 3), dtype=np.float32)
img = img/2
print(img.shape)
print(img[0][0])

img_tensor = transform1(img)
print(img_tensor.size())
print(img_tensor[:,0,0])

img_tensor_normal = transform2(img_tensor)
print(img_tensor_normal.size())
print(img_tensor_normal[:,0,0])

print("==============================================================")
img2 = np.random.random((3, 2, 2))
print(img2.shape)
print(img2)
print(img2.max())
print(img2.min())

img2_tensor = transform1(img2)
print(img2_tensor.size())
print(img2_tensor)
print(img2_tensor.max())
print(img2_tensor.min())

img2_tensor_normalize = transform2(img2_tensor)
print(img2_tensor_normalize.size())
print(img2_tensor_normalize)
print(img2_tensor_normalize.max())
print(img2_tensor_normalize.min())

