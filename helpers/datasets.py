import math
import pandas as pd
import numpy as np
import torchvision

from PIL import Image


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
        img = Image.open(self.image_paths[index])
        img.load()
        
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # remove alpha dimension from png
        img_tensor = img_tensor[:3,:,:]
        return img_tensor, np.array(self.labels[index])