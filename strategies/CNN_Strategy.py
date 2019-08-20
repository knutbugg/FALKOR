from strategies.Strategy import Strategy
import torch

class CNN_Strategy(Strategy):
	"""Abstract class representing a Strategy used by Gekko. The child class must create all NotImplemented methods"""

	def __init__(model, weights_path, image_path):
		"""Initialize CNN_Strategy instance"""
		self.model = model

		self.image_path = Path(image_path)

		# Load saved model weights from path
		self._load_weights(self.model, weights_path)

	def _load_weights(model, path):
		"""Load weights into model from path"""
		
		try:
			model.load_state_dict(torch.load(path))
		except:
			print('Saved model weights not found')

	def _save_model(model, path):
		"""Save a trained PyTorch model to path"""
		
		torch.save(model.state_dict(), path)	

	def _generate_chart_img(df):
		"""Take a df of ochl and save it as a fusion chart image at save_path"""
		chart_ti = ['sma20', 'macd', 'obv', 'bb20_low', 'bb20_mid', 'bb20_up']
		chart = Charting(df, col_label='time', row_label='close', tech_inds=chart_ti)
		chart.chart_to_image(self.image_path / 'most_recent.png')


	def _preprocess_df(df):
		"""Take an input df of ochlv data and return a df ready for creating a chart image with"""

		# Create Technical Indicators for df

		# price and volume as lists of floats
		price_list = [float(x) for x in df.close.tolist()]
		volume_list = [float(x) for x in df.volume.tolist()]

		sma20_list = sma(price_list, n=20)
		macd_list = macd(price_list)
		obv_list = obv(volume_list, price_list)

		bb20 = bollinger_bands(price_list, 20, mult=2)
		bb20_low = [x[0] for x in bb20]
		bb20_mid = [x[1] for x in bb20]
		bb20_up = [x[2] for x in bb20]

		ti_dict = {'sma20': sma20_list, 'macd': macd_list, 'obv': obv_list,
				 'bb20_low': bb20_low, 'bb20_mid': bb20_mid, 'bb20_up': bb20_up}

		# Cut all data to last 30 periods

		df = df.iloc[df.shape[0]-30:, :]

		for label, data in ti_dict.items():
			last_data = data[len(data)-30:]
			ti_dict[label] = last_data

			# add to df
			df[label] = last_data

		return df

	def _load_img_as_tensor(img_path):
		"""Returns tensor representation of image at img_path"""

		img = pil_image.open(img_path)
		img.load()
		
		img_tensor = torchvision.transforms.ToTensor()(img)
		
		# remove alpha dimension from png
		img_tensor = img_tensor[:3,:,:]
		return img_tensor

	def feed_data(recent_candles):
		"""Feed in a DataFrame of the last recent candles."""

		# add technical indicators to live_df
		input_df = self._preprocess_df(recent_candles)

		# generate image from input_df
		self._generate_chart_img(input_df)

	def predict():
		"""Returns prediction of price growth for the next t+5 candles"""

		# load created chart image as tensor
		img_tensor = self._load_img_as_tensor(self.image_path / "most_recent.png")

		# get buy/sell signal 
		self.model.eval() # set model to evaluation mode

		output = self.model(img_tensor)
		
		# if output[0] > 0, positive growth prediction

		return output[0]

	def update():
		"""Run whatever operations necessary to keep the strategy up-to-date with current data"""
		pass