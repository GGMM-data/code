import torch 
import random


class DynamicNet(torch.nn.Module):
   def __init__(self, D_in, H, D_out):
     super(DynamicNet, self).__init__()
     self.input_linear = torch.nn.Linear(D_in, H)
     self.hidden_linear = torch.nn.Linear(H, H)
     self.output_linear = torch.nn.Linear(H, D_out)

   def forward(self, x):
      x = self.input_linear(x).clamp(min=0)
      for _ in range(random.randint(0, 3)):
        x = self.hidden_linear(x).clamp(min=0)
      self.y_pred = self.output_linear(x)
      return self.y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)
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
