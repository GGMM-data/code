import torch

# contents
# 1.torch.tensor()
# 2.tensor.size()
# 3.torch.add()
# 4.tensor.unfold()
# 5.tensor.view()
# 6 tensor.item() to get the value of only one element tensor
# 7.tensor.numpy() and torch.tensor
# 8.cuda

#############################################################
# 1.how to create torch tensor

x = torch.rand(3,4)  # construct a 4x3 matrix, uniform [0,1] 
print(x)

x1 = torch.empty(5,3) # construct a 5x3 matrix, uninitialized
print(x1)

x2 = torch.randn(5,3) # construct a 5x3 matrix, normal distribution
print(x2)

x3 = torch.zeros(5,3) # construct a 5x3 matrix filled zeros
print(x3)

x4 = torch.zeros(5,3,4) # construct a 5x3x4 matrix filled zeros
print(x4)

x5 = torch.ones(5,3) # construct a 5x3 matrix filled ones
print(x5)

x6 = torch.ones(5,3,dtype=torch.long) # construct a tensor with dtype=torch.long
print(x6)

x7 = torch.tensor([1,2,3]) # construct a tensor direct from data
print(x7)

x8 = x.new_ones(5,4) # constuct a tensor has the same property as x
print(x8)

x9 = torch.full([4,3],9)  # construct a tensor with a value 
print(x9)

x10 = x.new_ones(5,4,dtype=torch.int) # construct a tensor with the same property as x, and also can have the specified type.
print(x10)

x11 = torch.randn_like(x,dtype=torch.float) # construct a tensor with the same shape with x, 

##########################################################
# 2.tensor.size()

x = torch.randn(4,3)
print(x.size())

##########################################################
# 3.torch.add()

x = torch.randn(4,3)
y = torch.randn(4,3)

#(1)
print(x+y)

#(2)
print(torch.add(x,y))

#(3)
result = torch.empty(4,3)
torch.add(x,y,out=result)
print(result)

#(4) in-place
y.add_(x)

###############
#in-place

x.copy_(y)
x.t_()

##############################
# 4.torch.unfold(dim, size, step)
# return a tensor which contaions all slices of size from self tensor in the dimension dim
#   dim : dimension in which unfolding happens
#   size : the size of each slice that is unfolded  
#   step : the step between each slice
print("======================================4.tensor.unfold()==================================\n")
x = torch.arange(0, 110, 1)
y = x.unfold(0, 100, 3)
print(x.size())
print(y.size())

###############
# 5.tensor.view()

print("======================================5.tensor.view()=====================================\n")
x = torch.randn(4, 3)
print("x.size()")
print("  ", x.size(),"\n")

y = x.view(12)
print("y = x.view(12)")
print("  ", y.size(),"\n")

z = x.view(-1, 2)
print("z = x.view(-1, 2)")
print("  ", z.size(),"\n")

w = x.view(-1)
print("w = x.view(-1)")
print("  ", w.size(),"\n")

u = x.view(-1, 1)
print("u = x.view(-1, 1)")
print("  ", u.size(),"\n")

v = x.view(1, -1)
print("v = x.view(1, -1)")
print("  ", v.size(),"\n")

print(x.size(), y.size(), z.size(), w.size(), u.size(), v.size())

###############
# 6 tensor.item() to get the value of only one element tensor

x = torch.randn(1)
print(x)
print(x.item())

###############
# 7.numpy and torch.tensor

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)


################
# 8.cuda
import numpy as np

a = np.ones([1,2])
b = torch.from_numpy(a)
print(b)
np.add(a,1,a)
print(a)
print(b)

try:
  if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device) # directly create a tensor on gpu
    x = x.to(device) # .to("cuda")
    z = x + y
    print(z) 
    print(z.to("cpu",torch.double)) # .to change dtype
except Exception as E:
    print(E)
print("End!")
