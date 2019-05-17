import torch
torch.manual_seed(5)
x = torch.empty(5, 3)
print(torch.empty(5, 3)) # construct a 5x3 matrix, uninitialized
print(torch.rand(3, 4))  # construct a 4x3 matrix, uniform [0,1] 
print(torch.randn(5, 3)) # construct a 5x3 matrix, normal distribution
print(torch.randn(2, 3).type())
print(torch.zeros(5, 3)) # construct a 5x3 matrix filled zeros
print(torch.ones(5, 3)) # construct a 5x3 matrix filled ones
print(torch.ones(5, 3, dtype=torch.long)) # construct a tensor with dtype=torch.long
print(torch.tensor([1,2,3])) # construct a tensor direct from data
print(x.new_ones(5,4)) # constuct a tensor has the same property as x
print(torch.full([4,3],9))  # construct a tensor with a value 
print(x.new_ones(5,4,dtype=torch.int)) # construct a tensor with the same property as x, and also can have the specified type.
print(torch.randn_like(x,dtype=torch.float)) # construct a tensor with the same shape with x, 
print(torch.ones_like(x))
print(torch.zeros_like(x))
print(torch.Tensor(3,4))
print(torch.Tensor(3,4).uniform_(0,1))
print(torch.Tensor(3,4).normal_(0,1))
print(torch.Tensor(3,4).fill_(5))
print(torch.arange(1, 3, 0.4))
