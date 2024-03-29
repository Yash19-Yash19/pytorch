import torch 

weights = torch.ones(4,requires_grad=True)

for epoch in range(3):
    model_outptut = (weights*3).sum()

    model_outptut.backward()

    print(weights.grad)

    weights.grad.zero_()



# another way using optimizer 
    
weights = torch.ones(4,requires_grad=True)

optimizer = torch.optim.SGD([weights], lr=0.01)
optimizer.step()
optimizer.zero_grad()