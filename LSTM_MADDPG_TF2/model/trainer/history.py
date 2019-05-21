import numpy as np


class History:
	def __init__(self, args, dims):
		self.cnn_format = args.cnn_format
		
		batch_size, history_length = args.batch_size, args.history_length
		
		self.history = np.zeros(
			[history_length] + list(dims), dtype=np.float32)
	
	def add(self, state):
		self.history[:-1] = self.history[1:]
		self.history[-1] = state
	
	def reset(self):
		self.history *= 0
	
	def obtain(self):
		if self.cnn_format == 'NHWC':
			return np.transpose(self.history, (1, 0))
		else:
			return self.history
