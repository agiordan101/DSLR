import numpy as np

# Logistic Regression One vs All
class Logreg():

	@classmethod
	def sigmoid(self, x):
		# return np.array([1 / (1 + np.exp(-x)) for x in np.nditer(inputs)])
		sig = 1 / (1 + np.exp(-x))
		return sig

	# @classmethod
	# def fa(inputs):
	# 	return np.array([sigmoid(x) for x in np.nditer(inputs)])


	def __init__(self, n_inputs, weights=None, lr=0.01):

		print(f"Logreg init(), weights={weights}")
		if weights:
			# Last weight = Bias
			self.w = np.array(weights[:-1])
			self.b = weights[-1]
		else:
			self.w = np.random.rand(n_inputs)
			self.b = np.random.rand()
		self.lr = lr


	def forward(self, inputs):
		# print(f"Forward inputs shape {inputs.shape}")
		# print(f"Forward weights shape {self.w.shape}")

		# if len(inputs.shape) > 1:
		# 	return np.array([self.sigmoid(np.dot(self.w, x) + self.b) for x in np.nditer(inputs)])
		# else:
		self.weighted_sum = np.dot(self.w, inputs) + self.b
		return self.sigmoid(self.weighted_sum)

	def get_batch(d1, d2, GD_batchsize=0.2):

		batchs = range(0, len(d1), int(GD_batchsize * len(d1)))
		batchs_min = batchs[:-1]
		batchs_max = batchs[1:]
		print(f"Batchs:{batchs_min}\n{batchs_max}")

		for bound_min, bound_max in zip(batchs_min, batchs_max):
			yield d1[bound_min:bound_max], d2[bound_min:bound_max]

	# def gradient_descent(train_x, train_y): # np.array m, nfeatures

	# 	depth_inv = 1 / len(train_x)		# int

	# 	A = self.forward(train_x)	# np.array m
	# 	fa = sigmoid(A)				# np.array m # Why use Activation and not weighted sum ???

	# 	# dA/dws = dA/dOut * dOut/dws
	# 	dA = (A - train_y) * fa * (1 - fa)

	# 	# Average gradient of each inputs series to get mean learning signal
	# 	# Cross product: dA/dw = dA/dws * dws/dw
	# 	# Not sure: We need to translate the matrix to iterate activation other each inputs series
	# 	dW = depth_inv * np.matmul(train_x, dA)
	# 	# dW = np.array([train_x for gradient in zip(train_x, dA.nditer())])		# np.array m, nfeatures
	# 	dB = depth_inv * np.sum(dA)

	# 	self.w = self.w - self.lr * dW
	# 	self.w = self.w - self.lr * dB

	def gradient_descent(features, target):

		prediction = self.forward(features)
		
		dloss = target - prediction
		dfa = self.sigmoid(self.weighted_sum) * (1 - self.sigmoid(self.weighted_sum))
		dws = features

		self.w = self.w - self.lr * dws * dfa * dloss
		self.b = self.b - self.lr * 1 * dfa * dloss

	# def backpropagation(self, train_x, train_y, GD_method='SGD'):
	# 	"""
	# 		dE/dw = dE/dOut * dOut/dws * dws/dw
	# 	"""
		# data = np.choices(zip(inputs, targets), len(inputs) / 4)

		# if GD_method == "SGD":

		# 	for input_batch, target_batch in get_batch(train_x, train_y):
		# 		self.gradient_descent(input_batch, target_batch)

		# else:
			# self.gradient_descent(inputs, targets)

	def save_weights(file_path):

		with open(file_path, 'w') as f:
			[f.write(f"{w}, ") for w in np.nditer(self.w)]
			f.write(f"{self.b}\n")