import numpy as np

class DataProcessing():

	def __init__(self, data, targets, columns=None):

		self.df = data
		self.targets = targets
		self.columns = columns
		self.normalization_data = []
		self.standardization_data = []

	def normalize(self):

		# new_lst = []
		data = {}

		if self.normalization_data:
			
			for item, data in zip(self.df.items(), self.normalization_data):
				data[item[0]] = [(x - _min) / (_max - _min) for x in column.values]

		else:
			for feature, column in self.df.items():
				_min = min(column)
				_max = max(column)
				self.normalization_data.append([_min, _max])
				data[feature] = [(x - _min) / (_max - _min) for x in column]

		self.df = data
		# self.df = pd.DataFrame(data=data, columns=self.columns)

	def get_data(self, data_type="2d_np_array", shuffle=True):

		if data_type == "2d_np_array":
			pass

		elif data_type == "2d_array":
			features = np.array([np.array(features) for features in zip(*list(self.df.values()))])
			targets = self.targets
			if shuffle:
				print(f"features {features.shape}:\n{features}")
				seed = np.random.get_state()
				np.random.shuffle(features)
				np.random.set_state(seed)
				np.random.shuffle(targets)
				
				print(f"features {features.shape}:\n{features}")
			return features, targets

		elif data_type == "DataFrame":
			return self.df

	def save_data(self, file_path, normalization=False, standardization=False):

		with open(file_path, 'w+') as f:

			if normalization:
				f.write("Normalization data\n")
				for _min, _max in self.normalization_data:
					f.write(f"{_min}/{_max}\n")
			
			if standardization:
				f.write("Standardization data\n")
				for mean, std in self.standardization_data:
					f.write(f"{mean}/{std}\n")

			f.close()

	def load_data(self, file_path, normalization=False, standardization=False):

		with open(file_path, 'r') as f:
			data = f.read()
			f.close()

			if normalization:
				self.normalization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
				print(f"normalization_data: {self.normalization_data}")

			if standardization:
				self.standardization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
				print(f"standardization_data: {self.standardization_data}")
