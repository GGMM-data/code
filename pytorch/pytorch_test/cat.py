import torch

# torch.cat(seq, dim=0, out=None) -> Tensor
# concat the given sequence of seq tensors in the give dimension.

x1 = torch.randn(3, 4)
x2 = torch.randn(1, 4)
x3 = torch.randn(8, 4)

x = torch.cat([x1,x2,x3])
print("x1 size:\n  ", x1.size())
print("x2 size:\n  ", x2.size())
print("x3 size:\n  ", x3.size())
print("\n  x = torch.cat([x1, x2, x3], 1)")
print("x size:\n  ", x.size())

print("\n\n")
y1 = torch.randn(3,4,5)
y2 = torch.randn(3,6,5)
y3 = torch.randn(3,1,5)

y = torch.cat([y1, y2, y3], 1)
print("y1 size:\n  ", y1.size())
print("y2 size:\n  ", y2.size())
print("y3 size:\n  ", y3.size())
print("\n  y = torch.cat([y1, y2, y3], 1)")
print("y size:\n  ", y.size())
