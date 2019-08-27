import torch
import torch.nn as nn
from torch.nn.functional import relu
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self):
        """Initialize the model by setting up the layers"""
        super(CNN, self).__init__()
        
        # initial layer is resnet
        self.resnet = models.resnet18(pretrained=True, progress=False)
        
        # final fully connected layers
        self.dense1 = nn.Linear(1000, 500)
        self.dense2 = nn.Linear(500, 100)
        self.dense3 = nn.Linear(100, 12)
        
        # output layer
        self.dense4 = nn.Linear(12, 1)
    
    def forward(self, x):
        """Perform a forward pass of our model on some input and hidden state"""
        
        x = self.resnet(x)
        
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
