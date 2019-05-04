import torch
import torch.nn.functional as F

N, D_in, H, D_out = 64, 1000, 100, 10

# define model
#model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out))
class my_model(torch.nn.Module):
   def __init__(self, D_in, H, D_out):
      super(my_model, self).__init__()
      self.linear1 = torch.nn.Linear(D_in, H)
      self.linear2 = torch.nn.Linear(H, D_out)

   def forward(self, x):
      x = self.linear1(x)
      relu = F.relu(x)
      y_pred = self.linear2(relu)
      return y_pred

model = my_model(D_in, H, D_out)
timesteps = 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# saving  model
loss_fn = torch.nn.MSELoss(reduction='sum')
lr = 1e-2
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

PATH = "./saving_model/"
for i in range(timesteps):
  y_pred = model(x)
  loss = loss_fn(y, y_pred)
  print(i, loss.item())
  model_path = PATH + "saving_example_step_" + str(i)
  torch.save(model.state_dict(), model_path)
  optimizer_path = PATH + " saving_example_state_" + str(i) + "_optimizer.pth"
  torch.save(optimizer.state_dict(), optimizer_path)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# loading model each step
model2 = my_model(D_in, H, D_out)
loss_fn2 = torch.nn.MSELoss(reduction='sum')

for i in range(timesteps):
  model2_path = PATH + "saving_example_step_" + str(i) 
  model2_state_dict = torch.load(model2_path)
  model2.load_state_dict(model2_state_dict)
#  model2.eval()
  y_pred = model2(x) 
  loss2 = loss_fn2(y, y_pred)
  print(i, loss2.item())
  #loss.backward()

# loading model and traing model
model3 = my_model(D_in, H, D_out)
loss_fn3 = torch.nn.MSELoss(reduction='sum')
optimizer3 = torch.optim.Adam(model3.parameters(), lr=lr)
for i in range(timesteps):
  if i < timesteps/2:
     model3_path = PATH + "saving_example_step_" + str(i)
     optimizer3_path = PATH + " saving_example_state_" + str(i) + "_optimizer.pth"
     model3_state_dict = torch.load(model3_path) 
     model3.load_state_dict(model3_state_dict)
     optimizer3_state_dict = torch.load(optimizer3_path)
     optimizer3.load_state_dict(optimizer3_state_dict)
#     model3.eval()
     y_pred = model3(x)
     loss3 = loss_fn3(y, y_pred)
     print(i, loss3.item())
#  elif i == timesteps/2:
#     model3.train()
  else:
     y_pred = model3(x)
     loss3 = loss_fn3(y, y_pred)
     print(i, loss3.item())
     optimizer3.zero_grad()
     loss3.backward()
     optimizer3.step()

