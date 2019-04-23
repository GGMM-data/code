import torch


class MyReLU(torch.autograd.Function):
    """ implement our custom function by subclassing torch.autograd.Function and implementing forward and backward passes which operate on Tensors
    """

    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input


dtype = torch.float
device = torch.device

N, D_in, H, D_out = 60, 1000, 100, 10

x = torch.randn(N, D_in, dtype=dtype, device=torch.device)
y = torch.randn(N, D_out, dtype=dtype, device=torch.device)

w1 = torch.randn(D_in, H, dtype=dtype, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=dtype, device=device, requires_grad=True)


learning_rate = 1e-6

for t in range(500):
    relu = MyReLU.apply

    # forward
    y_pred = relu(x.mm(w1).clamp(min=0).mm(w2))

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    #
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()


        
