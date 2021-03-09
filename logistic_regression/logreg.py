import numpy as np

# Logistic Regression One vs All
class Logreg():

	@classmethod
	def sigmoid(self, x):
		# return np.array([1 / (1 + np.exp(-x)) for x in np.nditer(inputs)])
		sig = 1 / (1 + np.exp(-x))
		print(f"sig({x}) = {sig}")
		return sig

	# @classmethod
	# def fa(inputs):
	# 	return np.array([sigmoid(x) for x in np.nditer(inputs)])


	def __init__(self, n_inputs, weights=None, lr=0.1, name=None):

		print(f"Logreg init(), {n_inputs} features, weights={weights}, name={name}")
		if weights:
			# Last weight = Bias
			self.w = np.array(weights[:-1])
			self.b = weights[-1]
		else:
			self.w = np.random.rand(n_inputs) * 2 - 1
			self.b = np.random.rand()
		print(f"\tWeight init: {self.w}")
		self.lr = lr
		self.name = name


	def forward(self, inputs):
		# print(f"Forward inputs shape {inputs.shape}")
		# print(f"Forward weights shape {self.w.shape}")

		# if len(inputs.shape) > 1:
		# 	return np.array([self.sigmoid(np.dot(self.w, x) + self.b) for x in np.nditer(inputs)])
		# else:
		self.weighted_sum = np.dot(self.w, inputs) + self.b
		return self.sigmoid(self.weighted_sum)

	def get_batch(self, d1, d2, GD_batchsize=0.2):

		batchs = range(0, len(d1), int(GD_batchsize * len(d1)))
		batchs_min = batchs[:-1]
		batchs_max = batchs[1:]
		print(f"Batchs:{batchs_min}\n{batchs_max}")

		for bound_min, bound_max in zip(batchs_min, batchs_max):
			yield d1[bound_min:bound_max], d2[bound_min:bound_max]

	def gradient_descent(self, features, target):

		prediction = self.forward(features)
		# print(f"Features: {features}")
		print(f"self.w: {self.w}") 
		print(f"Prediction: {prediction}")
		print(f"Target: {target}")

		dloss = prediction - target
		dfa = self.sigmoid(self.weighted_sum) * (1 - self.sigmoid(self.weighted_sum))
		dws = features

		# dE/dw = dE/dOut * dOut/dws * dws/dw
		self.w = self.w - self.lr * dws * dfa * dloss
		self.b = self.b - self.lr * 1 * dfa * dloss

		print(f"Loss: {dloss * dloss}")
		print(f"g1: {dloss}")
		print(f"g2: {dfa}")
		print(f"g3: {dws}")
		print(f"gradient: {dws * dfa * dloss}")
		print(f"self.w: {self.w}")

	# def backpropagation(self, train_x, train_y, GD_method='SGD'):
	# 	"""
	# 		dE/dw = dE/dOut * dOut/dws * dws/dw
	# 	"""
	# 	data = np.choices(zip(inputs, targets), len(inputs) / 4)

	# 	if GD_method == "SGD":

	# 		for input_batch, target_batch in get_batch(train_x, train_y):
	# 			self.gradient_descent(input_batch, target_batch)

	# 	else:
	# 		self.gradient_descent(inputs, targets)

	def save_weights(self, file_path):

		with open(file_path, 'w') as f:
			[f.write(f"{w}, ") for w in np.nditer(self.w)]
			f.write(f"{self.b}\n")
