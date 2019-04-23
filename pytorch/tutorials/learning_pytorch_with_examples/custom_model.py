import torch

class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)


  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss(reduction='sum')
lr = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for i in range(500):
  y_pred = model(x)
  loss = loss_fn(y, y_pred)
  print(i, loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
