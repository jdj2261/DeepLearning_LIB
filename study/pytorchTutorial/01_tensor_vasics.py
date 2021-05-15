import torch

x = torch.empty(1)  # scalar
print(x)

x = torch.empty(3)  # 1D vector
print(x)

x = torch.empty(2, 3) # matrix, 2D
print(x)

x = torch.empty(2,2,2,3)
print(x)

x = torch.rand(2, 2)
print(x)

x = torch.zeros(2, 2)
print(x)

x = torch.ones(2, 2, dtype=torch.float16)
print(x)

print(x.dtype)
print(x.size())

x = torch.tensor([2.5, 0.1])
print(x)

x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x + y
z1 = torch.add(x, y)
z2 = torch.sub(x, y)
z3 = torch.mul(x, y)
z4 = torch.div(x, y)

x = torch.rand(5, 3)
print(x)
print(x[1, :])

print(x[1,1]) # element at 1, 1

# Get the actual value if only 1 element in your tensor
print(x[1,1].item())

x = torch.rand(4,4)
print(x)
y = x.view(-1, 8)
print(y, y.size())

import numpy as np
a = torch.ones(5)
print(a)
b = a.numpy()
print(b, type(b))

# 수정할 때 조심
a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b, type(b))

# 수정할 때 조심
a += 1
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device = device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    
    # z.numpy() gpu 텐서에서 numpy 불가능
    z = z.to("cpu")
    z = z.numpy()
    print(z)
