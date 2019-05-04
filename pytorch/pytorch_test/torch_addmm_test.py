import torch
import numpy as np

mat = torch.ones(3,5)
mat1 = torch.ones(3,4)
mat2 = torch.ones(4,5)

m1 = mat1.numpy()
m2 = mat2.numpy()
m1 = np.mat(m1)
m2 = np.mat(m2)
print(m1*m2)
beta = 1
alpha = 1
result = torch.addmm(beta=beta, input=mat, alpha=alpha, mat1=mat1, mat2=mat2)
print(result)
