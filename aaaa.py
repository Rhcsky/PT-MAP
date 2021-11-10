import torch

# data = [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]
data = [[[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6]]]

data = torch.Tensor(data)
print(data.shape)
print(data)
print(-10 * data)
print(torch.exp(data))

# print(data.shape)
# print(data)
# print(data.mean(1))
# print(data.mean(2))
#
# print(data.norm(dim=1).shape)
# print(data.norm(dim=2).shape)
#
# print(data.norm(dim=1))
# print(data.norm(dim=2))

# import torch
#
# data = torch.ones((10, 5)) * 15
# print(data)
