import sys
from math import inf

import torch
from torch.nn import Sigmoid
from torch.autograd import Variable


def toyNetwork():
    # TODO: Implement network as given in the exercise sheet.
    # Torch documentation: https://pytorch.org/docs/stable/index.html

    # TODO: Define weight variables using: Variable(torch.Tensor(), requires_grad=True)
    x = torch.Tensor([1.0,1.0,1.0])
    y = 1

    #Weights
    w1 = 0.5
    w2 = .3
    w3 = 0.3
    w4 = 0.1
    w5 = 0.8
    w6 = 0.3
    w7 = Variable(torch.Tensor([0.5]), requires_grad=True)
    w8 = Variable(torch.Tensor([0.9]), requires_grad=True)
    w9 = Variable(torch.Tensor([0.2]), requires_grad=True)

    b1 = Variable(torch.Tensor([1.0]))
    b2 = Variable(torch.Tensor([1.0]))

    # TODO: Define network forward pass connectivity
    m = Sigmoid()
    error = inf
    lr = 0.01

    # TODO: Train network until convergence
    # Steps:
    # 1 - compute error
    # 2 - do backward propagation, use: error.backward() to do so
    # 3 - update weight variables according to gradient and learning rate
    # 4 - Zero weight gradients with w_.grad_zero_()

    while error > 0.001:
        w135 = Variable(torch.Tensor([w1,w3,w5]), requires_grad=True)
        w246 = Variable(torch.Tensor([w2,w4,w6]), requires_grad=True)
        h1 = m(w135.dot(x))
        h2 = m(w246.dot(x))
        y_hat = w7*h1+w8*h2+w9
        error = torch.sum( torch.pow(y - y_hat,2))
        error.backward()

        with torch.no_grad():
            w135 -= lr * w135.grad
            w246 -= lr * w246.grad
            w7 -= lr * w7.grad
            w8 -= lr * w8.grad
            w9 -= lr * w9.grad
            w135.grad.zero_()
            w246.grad.zero_()
            w7.grad.zero_()
            w8.grad.zero_()
            w9.grad.zero_()

        print("Error: {:.4f}".format(error))


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network toy example!")
    toyNetwork()
    print("Done!")
