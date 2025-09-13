import torch
a = torch.randn(1,3,4)
print(a.shape)
a = a.unsqueeze(0)
print(a.shape)
a = a.squeeze(0)
print(a.shape)