import torch
import numpy as np


# torch squeeze and unsqueeze

# torch.squeeze(input, dim=None, out=None) -> Tensor
# return a tensor with all the dimensions of input of size 1 removed
# torch.squeeze(input, dim, out=None) -> Tensor
# return a new tensor with a dimension of size one inserted at the speficied position
print("=================================================================")
print("torch.squeeze(input, dim=None, out=None)")
a = torch.ones([2, 1, 3, 1, 5])
print(a.size())

print("=================================================================\n")
print("torch.squeeze(a, 0):")
a0 = torch.squeeze(a, 0)
print("  ", a0.size())
print("torch.squeeze(a, 1):")
a1 = torch.squeeze(a, 1)
print("  ", a1.size())
print("torch.squeeze(a, 2):")
a2 = torch.squeeze(a, 2)
print("  ", a2.size())
print("torch.squeeze(a, 3):")
a3 = torch.squeeze(a, 3)
print("  ", a3.size())
print("torch.squeeze(a, 4):")
a4 = torch.squeeze(a, 4)
print("  ", a4.size())
print("torch.squeeze(a):")
a5 = torch.squeeze(a)
print("  ", a5.size())
print("==============================end================================\n")

print("=================================================================")
print("torch.unsqueeze(input, dim, out=None)")
b = torch.ones([3, 4, 5])
print(b.size())

print("=================================================================\n")

b0 = torch.unsqueeze(b, 0)
print(b0.size())
b1 = torch.unsqueeze(b, 1)
print(b1.size())
b2 = torch.unsqueeze(b, 2)
print(b2.size())
b3 = torch.unsqueeze(b, 3)
print(b3.size())
try:
  b4 = torch.unsqueeze(b, 4)
  print(b4.size())
except Exception as e:
  print("Error:", e)
try:
  b5 = torch.unsqueeze(b)
  print(b5.size())
except Exception as e:
  print("Error:", e)

print("==============================end================================\n")


print("=================================================================")
print("numpy.squeeze(a, axis=None)")
# numpy.squeeze() numpy.expand_dims()
a = np.ones([2, 3, 1, 4, 5])
print(a.shape)

print("=================================================================\n")

try:
  a0 = np.squeeze(a, 0)
  print(a0.shape)
except Exception as e:
  print("Error:", e)

try:
  a1 = np.squeeze(a, 1)
  print(a1.shape)
except Exception as e:
  print("Error:", e)

try:
  a2 = np.squeeze(a, 2)
  print(a2.shape)
except Exception as e:
  print("Error:", e)

try:
  a3 = np.squeeze(a, 3)
  print(a3.shape)
except Exception as e:
  print("Error:", e)

try:
  a4 = np.squeeze(a, 4)
  print(a4.shape)
except Exception as e:
  print("Error:", e)

print("==============================end================================\n")


print("=================================================================")
print("numpy.expand_dims(a, axis)")
b = np.ones([3, 4, 5])
print(b.shape)

print("=================================================================\n")
b0 = np.expand_dims(b, 0)
print(b0.shape)
b1 = np.expand_dims(b, 1)
print(b1.shape)
b2 = np.expand_dims(b, 2)
print(b2.shape)
b3 = np.expand_dims(b, 3)
print(b3.shape)
try:
  b4 = np.expand_dims(b, 4)
  print(b4.shape)
except Exception as e:
  print("Error:", e)

try:
  b5 = np.expand_dims(b, 5)
  print(b5.shape)
except Exception as e:
  print("Error:", e)

print("==============================end================================\n")


