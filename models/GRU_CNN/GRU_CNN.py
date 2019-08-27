import torch
import torch.nn as nn
from torch.nn.functional import relu
import torchvision.models as models


class GRU_CNN(nn.Module):
	def __init__(self, num_features, batch_size, hidden_size):
		"""Initialize the model by setting up the layers"""
		super(GRU_CNN, self).__init__()
		
		# initialize gru and cnn - the full models
		
		# gru model params
		self.num_features = num_features
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.n_layers = 1
		
		# resnet model
		self.cnn = models.resnet34(pretrained=True, progress=False)
			  
		# RNN-GRU model
		self.rnn = nn.GRU(input_size=self.num_features,
						  hidden_size=self.hidden_size)
		
		# init GRU hidden layer
		self.hidden = self.init_hidden(batch_size=self.batch_size, hidden_size=hidden_size)
		self.gru_output = nn.Linear(self.hidden_size, 1000)
		
		# final fully connected layers
		self.dense1 = nn.Linear(1000, 500)
		self.dense2 = nn.Linear(500, 100)
		self.dense3 = nn.Linear(100, 12)
		
		# output layer
		self.dense4 = nn.Linear(12, 1)
	
	def load_cnn_weights(self, cnn):
		cnn_params = cnn.named_parameters()
		gru_cnn_params = dict(self.cnn.named_parameters())
		
		for name, cnn_param in cnn_params:
			if name in gru_cnn_params:
				gru_cnn_params[name].data.copy_(cnn_param.data)
	
	def load_gru_weights(self, gru):
		gru_params = gru.named_parameters()
		gru_cnn_params = dict(self.rnn.named_parameters())
		
		for name, gru_param in gru_params:
			if name in gru_cnn_params:
				gru_cnn_params[name].data.copy_(gru_param.data)
	
	def forward(self, gru_input, cnn_input):
		"""Perform a forward pass of our model on some input and hidden state"""
  
		# gru
		gru_out, self.hidden = self.rnn(gru_input, self.hidden)
		
		# detatch the hidden layer to prevent further backpropagating. i.e. fix the vanishing gradient problem
		self.hidden = self.hidden.detach().cuda()
		
		# pass through linear layer
		gru_out = torch.squeeze(self.gru_output(gru_out))
				
		# cnn
		cnn_out = self.cnn(cnn_input)
		
		# add the outputs of grunet and cnn
		x = gru_out.add(cnn_out)
		
		# feed through final layers

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
