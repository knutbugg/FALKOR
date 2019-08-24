from helpers.technical_indicators import sma, macd, obv, bollinger_bands, ema

def split_dataset(dataset_df, a=0, b=30, step_size=5):
	"""Split dataset_df into a list of DataFrames. Sliding window of b-a. step_size = 5
	returns [ [DataFrame], [DataFrame], ..., [DataFrame] ] where length is dependent on a, b, and step_size. 
	"""
	
	end_b = dataset_df.shape[0] 
	
	dataset_splits = []
	
	# Take a 30 period window of dataset_df, with a step size of 5
	while b < end_b:

		window = dataset_df.iloc[a:b, :]
		dataset_splits.append(window)
		
		a += step_size
		b += step_size
	# dataset_splits = dataset_splits[:len(dataset_splits)-5] # remove last 5 element since we predict price t+5
	return dataset_splits

def price_labels(dataset_windows: list, period_size: int, periods_into_the_future=5):
	"""returns a list with len = len(dataset_windows) - 1 containing the price return of time + 5 and now"""
	dct = {'curr_price': [], 'future_price': [], 'return': []}
	for i, df in enumerate(dataset_windows[:-1]): # skip the last one
		curr_price = df['close'][period_size-1]
		dct['curr_price'].append(curr_price)
		
		future_price = dataset_windows[i+1]['close'][periods_into_the_future-1] # 4 periods into the future
		dct['future_price'].append(future_price) 
		dct['return'].append(( ( future_price - curr_price ) / curr_price) + 1) # add one s.t. log(return) > 0 if positive growt
		
	return dct

def add_ti(df):
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

	# Cut all data to have equal length

	smallest_len = min( [len(x) for l, x in ti_dict.items()] )

	df = df[start_i:]

	for label, data in ti_dict.items():

		# convert smallest_len to a start_index for this data list
		start_i = (len(data) - smallest_len)

		ti_dict[label] = data[start_i:]

		# add to df
		df[label] = last_data

	return df
