from strategies.Strategy import Strategy
from ../helpers.data_processing import add_ti
from ../helpers.charting_tools import Charting

import torch
import torchvision
from PIL import Image
from pathlib import Path

class CNN_Strategy(Strategy):
    """Abstract class representing a Strategy used by Gekko. The child class must create all NotImplemented methods"""

    def __init__(self, model, weights_path, image_path):
        """Initialize CNN_Strategy instance"""
        self.model = model

        self.image_path = Path(image_path)

        # Load saved model weights from path
        self._load_weights(self.model, weights_path)

    def _load_weights(self, model, path):
        """Load weights into model from path"""
        
        try:
            model.load_state_dict(torch.load(path))
        except:
            print('Saved model weights not found')

    def _save_model(self, model, path):
        """Save a trained PyTorch model to path"""
        
        torch.save(model.state_dict(), path)    

    def _generate_chart_img(self, df):
        """Take a df of ochl and save it as a fusion chart image at save_path"""
        chart_ti = ['sma20', 'macd', 'obv', 'bb20_low', 'bb20_mid', 'bb20_up']
        chart = Charting(df, col_label='time', row_label='close', tech_inds=chart_ti)
        chart.chart_to_image(self.image_path / 'most_recent.png')

    def _load_img_as_tensor(self, img_path):
        """Returns tensor representation of image at img_path"""

        img = Image.open(img_path)
        img.load()
        
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # remove alpha dimension from png
        img_tensor = img_tensor[:3,:,:]
        return img_tensor

    def feed_data(self, recent_candles):
        """Feed in a DataFrame of the last recent candles."""

        # add technical indicators to live_df
        input_df = add_ti(recent_candles)

        # generate image from input_df
        self._generate_chart_img(input_df)

    def predict(self):
        """Returns prediction of price growth for the next t+5 candles"""

        # load created chart image as tensor
        img_tensor = self._load_img_as_tensor(self.image_path / "most_recent.png")
        
        # add batch_dimension to img_tensor
        img_tensor = img_tensor.unsqueeze(0)

        # get buy/sell signal 
        self.model.eval() # set model to evaluation mode
        
        
        output = self.model(img_tensor)
        
        # if output[0] > 0, positive growth prediction
        if output[0] >= 0.0:
            return "buy"
        else:
            return "sell"

    def update(self):
        """Run whatever operations necessary to keep the strategy up-to-date with current data"""
        pass
