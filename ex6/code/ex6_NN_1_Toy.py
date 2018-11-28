import sys
import torch
from torch.autograd import Variable


def toyNetwork():
    # TODO: Implement network as given in the exercise sheet.
    # Torch documentation: https://pytorch.org/docs/stable/index.html

    # TODO: Define weight variables using: Variable(torch.Tensor(), requires_grad=True)

    # TODO: Define network forward pass connectivity

    # TODO: Train network until convergence
    # Steps:
    # 1 - compute error
    # 2 - do backward propagation, use: error.backward() to do so
    # 3 - update weight variables according to gradient and learning rate
    # 4 - Zero weight gradients with w_.grad_zero_()



if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network toy example!")
    toyNetwork()
    print("Done!")
