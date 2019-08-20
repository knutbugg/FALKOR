import multiprocessing
from joblib import Parallel, delayed
import math
import pandas as pd
import numpy as np
from PIL import Image as pil_image
import math
from os import listdir
from os.path import isfile, join
from tqdm import tqdm_notebook as tqdm
import requests
from datetime import datetime
from pathlib import Path
from IPython.display import clear_output

from strategy.TINNStrategy import prepare_data
from chartobjects.Charting import Charting

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import relu
from torch.backends import cudnn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torchvision
from torchvision import models

import warnings
warnings.filterwarnings("ignore")


def print_progress(at, total):
    """Clears cell output and prints percentage progress"""
    clear_output()
    print("progress: {}%".format(round((at/total)*100,2)))

def split_dataset(dataset_df, a=0, b=30, step_size=5):
    """Split dataset_df into a list of DataFrames. Sliding window of b-a. step_size = 5
    returns [ [DataFrame], [DataFrame], ..., [DataFrame] ] where length is dependent on a, b, and step_size. 
    """
    
    end_b = dataset_df.shape[0] 
    
    dataset_splits = []
    
    # Take a 30 period window of dataset_df, with a step size of 5
    while b < end_b:
        if b % 10000 == 0: print_progress(b, end_b)
        
        window = dataset_df.iloc[a:b, :]
        dataset_splits.append(window)
        
        a += step_size
        b += step_size
    # dataset_splits = dataset_splits[:len(dataset_splits)-5] # remove last 5 element since we predict price t+5
    return dataset_splits


def price_labels(dataset_windows, period_size):
    """returns a list with len = len(dataset_windows) - 1 containing the price return of time + 5 and now"""
    dct = {'curr_price': [], 'future_price': [], 'return': []}
    for i, df in enumerate(dataset_windows[:-1]): # skip the last one
        curr_price = df['close'][period_size-1]
        dct['curr_price'].append(curr_price)
        
        future_price = dataset_windows[i+1]['close'][4] # 4 periods into the future
        dct['future_price'].append(future_price) 
        dct['return'].append(( ( future_price - curr_price ) / curr_price) + 1) # add one s.t. log(return) > 0 if positive growt
        
    return dct     



def save_model(model, path):
    """Save a trained PyTorch model to path"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load weights into model from path"""
    try:
        model.load_state_dict(torch.load(path))
    except:
        print('file not found')

class ArrayTimeSeriesDataset(Dataset):
    """Dataset for historical timeseries data. 
    self.feature_dfs = [np.Array, np.Array, np.Array, ..., np.Array]
    self.labels = [np.Array, np.Array, np.Array, ..., np.Array]
    """
    def __init__(self, time_series, labels):
        self.time_series, self.labels = time_series, labels
        self.c = 1 # one label
    
    def __len__(self):
        return len(self.time_series)
    
    def __getitem__(self, i):
        time_series_arr =  np.array(self.time_series[i])
        label = np.array(self.labels[i])
        return time_series_arr.flatten(), label.flatten() # convert array into vector

class ChartImageDataset(Dataset):
    """Stock chart image Dataset"""
    
    def __init__(self, image_paths: list, labels: list):
        """ 
        image_paths: list containing path to image. Order is maintained
        labels: list containing label for each image
        """
        self.image_paths = image_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, index):
        """Return Tensor representation of image at images_paths[index]"""
        img = pil_image.open(self.image_paths[index])
        img.load()
        
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # remove alpha dimension from png
        img_tensor = img_tensor[:3,:,:]
        return img_tensor, np.array(self.labels[index])

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
        
#       # flatten the output from GRU layer
#       x = x.contiguous().view(-1, self.hidden_size)
        
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


class ST_CNN(nn.Module):
    def __init__(self):
        """Initialize the model by setting up the layers"""
        super(ST_CNN, self).__init__()
        
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
        self.cnn = models.resnet18(pretrained=True, progress=False)
              
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



def train(model, num_epochs, batch_size, train_gen, valid_gen, test_gen, gru=False):
    """Standard training function used by all three models"""
    
    # For optimizing our model, we choose SGD 
    optimizer = optim.Adam(model.parameters(), lr=1e-4,)
    
    # training loop
    
    # toop through the dataset num_epoch times
    for epoch in range(num_epochs):
               
        # train loop
        train_loss = []
        valid_loss = []
        
        # take the batch and labels for batch 
        for batch, labels in train_gen:
            
            if gru:
                # add extra dimension to every vector in batch
                batch.unsqueeze_(-1)
                batch = batch.expand(batch.shape[0], batch.shape[1], 1)
                
                # reformat dimensions
                batch = batch.transpose(2,0)
                batch = batch.transpose(1, 2)
                
            batch, labels = batch.cuda(), labels.cuda()
            batch, labels = batch.float(), labels.float()
            
            # clear gradients
            model.zero_grad()
            output = model(batch)
            
            if gru:
                output = output[0] # turn (1, batch_size, 1) to (batch_size, 1)
            
            # declare the loss function and calculate output loss
            
            # we use the RMSE error function to train our model
            criterion = nn.MSELoss()
            
            loss = torch.sqrt(criterion(output, labels))
            
            # backpropogate loss through model
            loss.backward()

            # perform model training based on propogated loss
            optimizer.step()
            
            train_loss.append(loss)
        
        # validation loop
        
        profit = 0
        with torch.set_grad_enabled(False):
            for batch, labels in valid_gen:
                if gru:
                    # add extra dimension to every vector in batch
                    batch.unsqueeze_(-1)
                    batch = batch.expand(batch.shape[0], batch.shape[1], 1)

                    # reformat dimensions
                    batch = batch.transpose(2,0)
                    batch = batch.transpose(1, 2)
                    
                batch, labels = batch.cuda(), labels.cuda()
                batch, labels = batch.float(), labels.float()
                
                # transform the model from training configuration to testing configuration. ex. dropout layers are removed
                model.eval()

                output = model(batch)
                
                if gru:
                    output = output[0] # turn (1, batch_size, 1) to (batch_size, 1)
                
                val_loss = torch.sqrt(criterion(output, labels))
                
                model.train()
                
                valid_loss.append(val_loss)
                
            
            # Profitability testing
            profit = 0.0
            
            for batch, labels in test_gen:
                if gru:
                    # add extra dimension to every vector in batch
                    batch.unsqueeze_(-1)
                    batch = batch.expand(batch.shape[0], batch.shape[1], 1)

                    # reformat dimensions
                    batch = batch.transpose(2,0)
                    batch = batch.transpose(1, 2)
                
                batch, labels = batch.cuda(), labels.cuda()
                batch, labels = batch.float(), labels.float()
                
                # transform the model from training configuration to testing configuration. ex. dropout layers are removed
                model.eval()
                
                output = model(batch)
                
                if gru:
                    output = output[0] # turn (1, batch_size, 1) to (batch_size, 1)
                
                # if output is > 0 ==> model predict positive growth for the next five cycles. Purchase now and sell in 5 periods.
                for i, pred in enumerate(output):
                    if pred[0] > 1: # price will increase
                        profit += labels[i]
                       
                model.train()
                
                
        print("Epoch: {}/{}...".format(epoch+1, num_epochs),
              "Training Loss: {}".format(round(float(sum(train_loss)/len(train_loss)), 4)),
              "Validation Loss: {}".format(round(float(sum(valid_loss)/len(valid_loss)), 4)),
              "Profitability: {}".format(round(float(profit), 3)))     



def train_dual(model, num_epochs, batch_size, train_gen1, train_gen2, valid_gen1, valid_gen2, test_gen, gru=False):
    """Standard training function used by all three models"""
    # For optimizing our model, we choose SGD 
    optimizer = optim.Adam(model.parameters(), lr=1e-4,)
    
    # training loop
    
    # toop through the dataset num_epoch times
    for epoch in range(num_epochs):
        
        # train loop
        
        train_loss = []
        valid_loss = []
        
        # loop through each batch
        for i  in range(batch_size):
            gru_batch, gru_labels = next(iter(train_gen1))
            gru_batch, gru_labels = gru_batch.cuda(), gru_labels.cuda()
            gru_batch, gru_labels = gru_batch.float(), gru_labels.float()
            
            # add extra dimension to every vector in batch
            gru_batch.unsqueeze_(-1)
            gru_batch = gru_batch.expand(gru_batch.shape[0], gru_batch.shape[1], 1)

            # reformat dimensions
            gru_batch = gru_batch.transpose(2,0)
            gru_batch = gru_batch.transpose(1, 2)
            cnn_batch, cnn_labels = next(iter(train_gen2))
            cnn_batch, cnn_labels = cnn_batch.cuda(), cnn_labels.cuda()
            cnn_batch, cnn_labels = cnn_batch.float(), cnn_labels.float()
            
            # clear gradients
            model.zero_grad()
            output = model(gru_batch, cnn_batch)
            output = output[0]
            # declare the loss function and calculate output loss
            
            # we use the RMSE error function to train our model
            criterion = nn.MSELoss()
            
            loss = torch.sqrt(criterion(output, gru_labels))
            
            # backpropogate loss through model
            loss.backward()
            # perform model training based on propogated loss
            optimizer.step()
            
            train_loss.append(loss)
        
        
        # validation loop
        with torch.set_grad_enabled(False):
            for i in range(batch_size):
                gru_batch, gru_labels = next(iter(valid_gen1))
                gru_batch, gru_labels = gru_batch.cuda(), gru_labels.cuda()
                gru_batch, gru_labels = gru_batch.float(), gru_labels.float()
                
                # add extra dimension to every vector in batch
                gru_batch.unsqueeze_(-1)
                gru_batch = gru_batch.expand(gru_batch.shape[0], gru_batch.shape[1], 1)

                # reformat dimensions
                gru_batch = gru_batch.transpose(2,0)
                gru_batch = gru_batch.transpose(1, 2)

                cnn_batch, cnn_labels = next(iter(valid_gen2))
                cnn_batch, cnn_labels = cnn_batch.cuda(), cnn_labels.cuda()
                cnn_batch, cnn_labels = cnn_batch.float(), cnn_labels.float()
                
                # transform the model from training configuration to testing configuration. ex. dropout layers are removed
                model.eval()

                output = model(gru_batch, cnn_batch)
                output = output[0]
                
                val_loss = torch.sqrt(criterion(output, gru_labels))
                
                model.train()
                
                valid_loss.append(val_loss)
                
             # Profitability testing
            profit = 0.0
            
            for batch, labels in test_gen:
                gru_batch, gru_labels = next(iter(valid_gen1))
                gru_batch, gru_labels = gru_batch.cuda(), gru_labels.cuda()
                gru_batch, gru_labels = gru_batch.float(), gru_labels.float()
                
                # add extra dimension to every vector in batch
                gru_batch.unsqueeze_(-1)
                gru_batch = gru_batch.expand(gru_batch.shape[0], gru_batch.shape[1], 1)

                # reformat dimensions
                gru_batch = gru_batch.transpose(2,0)
                gru_batch = gru_batch.transpose(1, 2)

                cnn_batch, cnn_labels = next(iter(valid_gen2))
                cnn_batch, cnn_labels = cnn_batch.cuda(), cnn_labels.cuda()
                cnn_batch, cnn_labels = cnn_batch.float(), cnn_labels.float()
                
                # transform the model from training configuration to testing configuration. ex. dropout layers are removed
                model.eval()

                output = model(gru_batch, cnn_batch)
                output = output[0]
                
                # if output is > 0 ==> model predict positive growth for the next five cycles. Purchase now and sell in 5 periods.
                
                for i, pred in enumerate(output):
                    if pred > 1: # price will increase
                        profit += labels[i]
                
                
                model.train()
                
        print("Epoch: {}/{}...".format(epoch+1, num_epochs),
              "Training Loss: {}".format(round(float(sum(train_loss)/len(train_loss)), 4)),
              "Validation Loss: {}".format(round(float(sum(valid_loss)/len(valid_loss)), 4)),
              "Profitability: {}".format(round(float(profit), 3))) 


import os

path = 'data/Stocks'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

for file in files:
    print("training on: {}".format(file))
    dataset_df = pd.read_csv(file)
    dataset_df.reset_index(drop=True, inplace=True)
    dataset_df = dataset_df.drop('OpenInt', axis=1)
    dataset_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    timestamp_col = [datetime.timestamp(datetime.strptime(dte, '%Y-%m-%d')) for dte in dataset_df.date]
    dataset_df.date = timestamp_col
    dataset_df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    #dataset_df.to_csv('data/Stocks/aapl.csv', index=False)
    
    if dataset_df.shape[0] < 3000: 
        print("not enough data for this stock. ")
        continue


    # add tech ind
    dataset_df = prepare_data(dataset_df)
    dataset_df['return'] = np.log(dataset_df.close) - np.log(dataset_df.close.shift(1))
    print("shape of dataset_df: {}".format(dataset_df.shape))
    
    chart_ti = ['sma10', 'bb10_low', 'bb10_mid', 'bb10_up', 'bb20_low', 'bb20_mid', 'bb20_up'] 

    # Split dataset_df into slices split by window size of 30 and step_size 5
    dataset_windows = split_dataset(dataset_df, a=0, b=30, step_size=5)

    for df in dataset_windows: df.reset_index(drop=True, inplace=True) # reindex 0-29
       
    # Paths to training images and testing images

    train_img_path = Path('data/training-images/')
    test_img_path = Path('data/testing-images/')

    image_path_list = [train_img_path / 'image-{}.png'.format(i) for i in range(len(dataset_windows))]

    def generate_chart_image(i):
        df = dataset_windows[i] # grab the i'th window of the df
        chart = Charting(df=df, col_label='time', row_label='close', tech_inds=chart_ti)
        chart.chart_to_image(train_img_path / 'image-{}.png'.format(i)) # the / is a Path join method 

    # save every DataFrame of price + vol + tech. id. data into a chart image
    p = multiprocessing.Pool(processes = 7)

    p.map_async(generate_chart_image, [i for i in range(len(dataset_windows))])

    p.close()
    p.join()



    price_labels_dct = price_labels(dataset_windows, period_size=30)
    price_returns = price_labels_dct['return']
    # ensure len(dataset_windows) == len(labels_for_windows)
    dataset_windows = dataset_windows[:len(price_returns)]


    def normalize_data(i):
        df = dataset_windows[i]
        #if 'open' in df: df = df.drop('open', axis=1)
        #if 'close' in df: df = df.drop('close', axis=1)
        #if 'low' in df: df = df.drop('low', axis=1)
        #if 'high' in df: df = df.drop('high', axis=1)
        if 'time' in df: df = df.drop('time', axis=1)
        
        dataset_windows[i] = np.log(df + 1).fillna(0) # replace all NaN with 0 
        #dataset_windows[i] = df.fillna(0)

    for i in range(len(dataset_windows)): normalize_data(i)


    dataset_windows = dataset_windows[1:]
    price_returns = price_returns[1:] 
    curr_prices = price_labels_dct['curr_price'][1:]
    future_prices = price_labels_dct['future_price'][1:]


    # Training

    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware
    cudnn.benchmark = True

    # Parameters
    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 5}

    num_epochs = 5

    while len(dataset_windows) % params['batch_size'] != 0:
        dataset_windows = dataset_windows[:-1]
        
    while len(curr_prices) % params['batch_size'] != 0:
        curr_prices = curr_prices[:-1]

    while len(future_prices) % params['batch_size'] != 0:
        future_prices = future_prices[:-1]

    while len(price_returns) % params['batch_size'] != 0:
        price_returns = price_returns[:-1]
      
    assert len(dataset_windows) == len(price_returns) == len(curr_prices) == len(future_prices)
               

    # cnn

    # specify the split between train_df and valid_df from the process of splitting dataset_windows and labels_for_windows
    split = 0.7

    s = int(len(dataset_windows) * 0.7)
    while s % params['batch_size'] != 0:
        s += 1

    # create two ChartImageDatasets, split by split, for the purpose of creating a DataLoader for the specific model

    train_ds_cnn = ChartImageDataset(image_path_list[:s], price_returns[:s])
    valid_ds_cnn = ChartImageDataset(image_path_list[s:], price_returns[s:])

    # add potential profit as label
    test_ds_cnn = ChartImageDataset(image_path_list[s:], [future_prices[i] - curr_prices[i] for i in range(s, len(future_prices))])

    train_gen_cnn = DataLoader(train_ds_cnn, **params)
    valid_gen_cnn = DataLoader(valid_ds_cnn, **params)
    train_gen_cnn = DataLoader(valid_ds_cnn, **params)
    cnn = ST_CNN().cuda()

    cnn_path = Path('strategy/cnn/cnn_weights')
    load_model(cnn, cnn_path)

    train(cnn, num_epochs, batch_size=params['batch_size'], train_gen=train_gen_cnn, valid_gen=valid_gen_cnn, test_gen=train_gen_cnn)
    save_model(cnn, cnn_path)


    # gru
    train_ds_gru = ArrayTimeSeriesDataset(dataset_windows[:s], price_returns[:s])
    valid_ds_gru = ArrayTimeSeriesDataset(dataset_windows[s:], price_returns[s:])
    test_ds_gru = ArrayTimeSeriesDataset(dataset_windows[s:], [future_prices[i] - curr_prices[i] for i in range(s, len(future_prices))])

    hidden_size = 800

    train_gen_gru = DataLoader(train_ds_gru, **params)
    valid_gen_gru = DataLoader(valid_ds_gru, **params)
    test_gen_gru = DataLoader(valid_ds_gru, **params)

    gru = GRUnet(num_features=390, batch_size=params['batch_size'], hidden_size=hidden_size).float().cuda()

    gru_path = Path('strategy/gru/gru_weights')
    load_model(gru, gru_path)
    train(gru, num_epochs, batch_size=params['batch_size'], train_gen=train_gen_gru, valid_gen=valid_gen_gru, test_gen=test_gen_gru, gru=True)
    save_model(gru, gru_path)

    # gru-cnn
    gru_cnn = GRU_CNN(num_features=390, batch_size=params['batch_size'], hidden_size=800).float().cuda()
    gru_cnn_path = Path('strategy/cnn-gru/cnn_gru_weights')
    load_model(gru_cnn, gru_cnn_path)

    gru_cnn.load_cnn_weights(cnn)
    gru_cnn.load_gru_weights(gru)

    train_dual(gru_cnn, num_epochs, batch_size=params['batch_size'], train_gen1=train_gen_gru, train_gen2=train_gen_cnn,
               valid_gen1=valid_gen_gru, valid_gen2=valid_gen_cnn, test_gen=test_gen_gru, gru=True)

    save_model(gru_cnn, gru_cnn_path)

