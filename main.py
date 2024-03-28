import torch
import numpy as np 

# creating tensor
x = torch.empty(2)
print(x)


# creating random tensor
x = torch.rand(2, 2)
print(x)

# creating tensor filled with zeros
x = torch.zeros(2, 2)
print(x)

# creating tensor filled with ones
x = torch.ones(2, 2)
print(x)

# creating tensor from data
x = torch.tensor([2.5, 0.1])
print(x)


x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x+y
# z=torch.add(x, y)   # another way to add
print(z)


z = torch.sub(x, y)   # subtract
# z=x - y
print(z)

z = torch.mul(x, y)   # multiply
# z=x * y
print(z)

z = torch.div(x, y)   # divide
# z=x / y
print(z)

# slicing
x = torch.rand(5, 3)
print(x)
print(x[:, 0])  # all rows and first column
print(x[1, :])  # second row and all columns
print(x[1, 1].item())

# reshaping tensor
x=torch.rand(4,4)
print(x)
y=x.view(16)    
print(y)
z=x.view(-1,8)  # -1 means automatically calculate the number of rows
print(z)
print(z.size())


# converting tensor to numpy
a=torch.ones(5)
print(a)
b=a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# converting numpy to tensor
a=np.ones(5)
b=torch.from_numpy(a)
print(a)
print(b)

np.add(a, 1, out=a)
print(a)
print(b)

np.subtract(a, 1, out=a)
print(a)
print(b)

np.multiply(a, 1, out=a)
print(a)
print(b)

np.divide(a, 1, out=a)
print(a)
print(b)

np.sqrt(a, out=a)
print(a)
print(b)



if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)
    z = z.to("cpu", torch.double)
    print(z)
    print(z.dtype)
    print(z.device)


x=torch.ones(5, requires_grad=True)
print(x)
