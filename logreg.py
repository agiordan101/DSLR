import numpy as np

# class Neuron():

# 	def __init__(self, n_inputs, weights=None):

# 		if weights:
# 			self.w = np.array(weights[:-1])
# 			self.b = weights[-1]
# 		else:
# 			self.w = np.random.rand(n_inputs)
# 			self.b = np.random.rand()
	
# 	def forward(self, inputs):
# 		return np.dot(self.w, inputs) + self.b

# 	def backpropagation(self, errors):
# 		pass


# class Softmax():

# 	def __init__(self, n_inputs):
# 		pass
	
# 	def forward(self, inputs)
# 		pass
	
# 	def backpropagation(self, errors):
# 		pass


# Logistic Regression One vs All
class Logreg():

	@classmethod
	def sigmoid(self, inputs):
		# print(x)
		return np.array([1 / (1 + np.exp(-x)) for x in np.nditer(inputs)])

	@classmethod
	def fa(inputs):
		return np.array([sigmoid(x) for x in np.nditer(inputs)])


	def __init__(self, n_inputs, n_outputs, weights=None):

		if weights:
			self.w = np.array(weights[:-1])
			self.b = weights[-1]
		else:
			self.w = np.random.rand(n_inputs)
			self.b = np.random.rand()


	def forward(self, inputs)
		return self.sigmoid(np.dot(self.w, inputs) + self.b)

	def backpropagation(self, inputs, targets):
		
		data = np.choices(zip(inputs, targets), len(inputs) / 4)

		outputs = self.forward(inputs)
		dE = 
