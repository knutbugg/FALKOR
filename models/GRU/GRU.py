import torch
import torch.nn as nn
from torch.nn.functional import relu
import torchvision.models as models


class GRUnet(nn.Module):
	def __init__(self, num_features, batch_size, hidden_size):
		"""Initialize the model by setting up the layers"""
		super(GRUnet, self).__init__()
		
		# initialize information about model
		self.num_features = num_features
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.n_layers = 1
		
		# RNN-GRU Layer
		self.rnn = nn.GRU(input_size=self.num_features,
						  hidden_size=self.hidden_size)
		
		# init GRU hidden layer
		self.hidden = self.init_hidden(batch_size=self.batch_size, hidden_size=hidden_size)
		
		# dropout layer
		#self.dropout = nn.Dropout(0.3)
		
		# 3 fully-connected hidden layers - with an output of dim 1
		self.link_layer = nn.Linear(self.hidden_size, 1000)
		self.dense1 = nn.Linear(1000, 500)
		self.dense2 = nn.Linear(500, 100)
		self.dense3 = nn.Linear(100, 12)
		
		# output layer
		self.dense4 = nn.Linear(12, 1)
		
	def forward(self, x):
		"""Perform a forward pass of our model on some input and hidden state"""
		# GRU layer
		x, self.hidden = self.rnn(x, self.hidden)
		
		# detatch the hidden layer to prevent further backpropagating. i.e. fix the vanishing gradient problem
		self.hidden = self.hidden.detach().cuda()
				
		# apply a Dropout layer 
		#x = self.dropout(x)
		
		# pass through the link_layer
		x = self.link_layer(x)
		x = relu(x)
		
		# apply three fully-connected Linear layers with ReLU activation function
		x = self.dense1(x)
		x = relu(x)
		
		x = self.dense2(x)
		x = relu(x)
		
		x = self.dense3(x)
		x = relu(x)
		
		# output is a size 1 Tensor
		x = self.dense4(x)
		return x
	
	def init_hidden(self, batch_size, hidden_size):
		"""Initializes hidden state"""
		
		# Creates initial hidden state for GRU of zeroes
		hidden = torch.ones(1, self.batch_size, hidden_size).cuda()
		return hidden
