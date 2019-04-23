import torch

# data
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out),)
print(model.parameters())

loss_fn = torch.nn.MSELoss()

lr = 1e-4

for i in range(500):
  y_pred = model(x)
  loss = loss_fn(y, y_pred)
  #print(i, loss.item())
  model.zero_grad()
  loss.backward()
  with torch.no_grad():
    for param in model.parameters():
      param -= param.grad*lr
    
