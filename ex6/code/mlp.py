import torch
import torch.nn as nn

class MLP(nn.Module):
	'''
	Multi-layer perceptron class
	Define the Neural network architecture
	'''
	def __init__(self,input_shape):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_shape[1], 2)
		self.fc3 = nn.Linear(2, 2)

	def forward(self,W):
		yf1 = torch.sigmoid(self.fc1(W))
		y = self.fc3(yf1)
		return y
