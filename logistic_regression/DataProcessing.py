import numpy as np
import pandas as pd

class DataProcessing():

	def __init__(self, data, column=None):

		if not isinstance(data, pd.DataFrame):
			data = pd.DataFrame(data=data, column=column)
		self.df = data
		self.column = column
		self.normalization_data = []
		self.standardization_data = []

	def normalize():

		new_lst = []

		if self.normalization_data:
			
			for item, data in zip(self.df.items(), self.normalization_data):
				new_lst.append([(x - data[0]) / (data[1] - data[0]) for x in item[1].values])

		else:
			for feature, column in self.df.items():
				_min = column.min
				_max = column.max
				self.normalization_data.append([_min, _max])
				new_lst.append([(x - _min) / (_max - _min) for x in column.values])

		self.df = pd.DataFrame(data=new_lst, column=self.column)

	def save_data(file_path, normalization=False, standardization=False):

		with open(file_path, 'w') as f:

			if normalization:
				f.write("Normalization data")
				for _min, _max in self.normalization_data:
					f.write(f"{_min}/{_max}")
			
			if standardization:
				f.write("Standardization data")
				for mean, std in self.standardization_data:
					f.write(f"{mean}/{std}")

			f.close()

	def load_data(file_path, normalization=False, standardization=False):

		with open(file_path, 'r') as f:
			data = f.read()
			f.close()

			if normalization:
				self.normalization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
				print(f"normalization_data: {self.normalization_data}")

			if standardization:
				self.standardization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
				print(f"standardization_data: {self.standardization_data}")
