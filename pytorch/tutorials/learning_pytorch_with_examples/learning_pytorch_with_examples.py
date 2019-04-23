import torch
import torch.nn as nn
import numpy as np


##################
# Contents
#   Warm-up: numpy


#########################
#   Warm-up: numpy
N, D_in, H, D_out  = 64, 1000, 100, 10 # batch size, input dimension, hidden dimension, output dimension
epoch_nums = 500

# create input and output
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

for i in range(epoch_nums):
    # forward
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # compute loss
    loss = np.square(y_pred - y).sum()
    print(i, loss)

    # backward
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)


    # update weights
    w1 = w1 - learning_rate * grad_w1
    w2 -= learning_rate * grad_w2


import torch

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)
print(x.requires_grad)
w1 = torch.randn(D_in, H,device=device,dtype=dtype,requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # loss is a Tensor of shape (1,), loss.item() gets the scalar value
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())  # item() to get the value of a (1,) tensor

    loss.backward()

    # update weights manually
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_() # manually zero the gradients after updating weigths
        w2.grad.zero_()

    # also you can update the weight automatically
    #
    # import torch.optim as optim
    # optimizer = optim.SGD(net.parameters(), learning_rate=learning_rate)
    # optimizer.step()



