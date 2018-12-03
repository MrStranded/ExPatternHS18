import torch
from torch.autograd import Variable

x = torch.Tensor(range(-5,5))
y = 3*x + 4

w = Variable(torch.Tensor([1.0]), requires_grad=True)
b = Variable(torch.Tensor([1.0]), requires_grad=True)

lr = 0.01

for i in range(25):
	y_hat = w*x + b

	error = torch.sum( torch.pow(y - y_hat,2) )
	error.backward()

	# update parameters
	with torch.no_grad():
		w -= lr * w.grad
		b -= lr * b.grad
		w.grad.zero_()
		b.grad.zero_()
	print("Error: {:.4f}".format(error))

print("w_pred = %.2f, b_pred = %.2f" % (w, b))