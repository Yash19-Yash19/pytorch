import torch

x = torch.randn(3)
x = torch.randn(3, requires_grad=True)
# y = torch.randn(3,requires_grad=True)
print(x)
# print(y)


y = x + 2
print(y)
z = y*y + 2
z = z.mean()
print(z)

# z.backward()
# print(x.grad)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
z.backward()
print(x.grad)

# x.require_grad_grad()
# x.detach()
# with torch.no_grad():
y = x.detach()
print(y)

with torch.no_grad():
    y = x + 2
    print(y)
