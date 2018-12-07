import torch
import torch.nn as nn

class MLP(nn.Module):
	'''
	Multi-layer perceptron class
	Define the Neural network architecture
	'''
	def __init__(self,input_shape):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_shape[1], 3)
		self.fc2 = nn.Linear(3, 3)
		self.fc3 = nn.Linear(3, 2)

	def forward(self,W):
		yf1 = torch.sigmoid(self.fc1(W))
		yf2 = torch.sigmoid(yf1)
		y = self.fc3(yf2)
		return y

# standard: 78-80% parameters: without bias: 8 with bias: 12
# 6 hidden neurons: 80-82% parameters: without bias: 24  with bias: 32
# 10 hidden neurons: 81-83% parameters: without bias: 40  with bias: 52
# 2 hidden layers à 3 neurons: 80-81% (little bit lower than 1 hidden layer with 6 neurons) parameters: without bias: 21 with bias: 29