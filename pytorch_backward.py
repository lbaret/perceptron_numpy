import torch

a = torch.tensor([2.0, 4.0], requires_grad=True)

b = torch.tensor([[1.0, 5.0], [6.0, 7.0]], requires_grad=True)

b

c = torch.matmul(a, b)

c

c.backward(torch.FloatTensor([0, 1]), retain_graph=True)

a.grad.data

b.grad.data

a.grad.data.zero_()
b.grad.data.zero_()

c.backward(torch.FloatTensor([1, 0]), retain_graph=True)

a.grad.data

b.grad.data

a.grad.data.zero_()
b.grad.data.zero_()

c.backward(torch.FloatTensor([1, 1]), retain_graph=True)

a.grad.data

b.grad.data

a.grad.data.zero_()
b.grad.data.zero_()


