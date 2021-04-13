import math
import numpy as np

house_matrix = ["Gryffindor",
				"Hufflepuff",
				"Ravenclaw",
				"Slytherin"]

pertinent_features = {
	'Arithmancy': 6,
	'Astronomy': 7,
	'Herbology': 8,
	'Defense Against the Dark Arts': 9,
	'Divination': 10,
	'Muggle Studies': 11,
	'Ancient Runes': 12,
	'History of Magic': 13,
	'Transfiguration': 14,
	'Potions': 15,
	'Care of Magical Creatures': 16,
	'Charms': 17,
	'Flying': 18
}

# pertinent_features = {
# 	'Astronomy': 7,
# 	'Herbology': 8,
# 	'Muggle Studies': 11,
# 	'History of Magic': 13,
# }

columns_name = list(pertinent_features.keys())

# Logistic Regression One vs All
class Logreg():

	@classmethod
	def sigmoid(self, x):
		sig = 1 / (1 + np.exp(-x))
		return sig

	def __init__(self, n_inputs, weights=None, lr=0.5, name=None):

		print(f"Logreg init(), {n_inputs} features, weights={weights}, name={name}")
		if weights:
			# Last weight = Bias
			self.w = np.array(weights[:-1])
			self.b = weights[-1]
		else:
			self.w = np.random.rand(n_inputs) * 2 - 1
			self.b = np.random.rand()
		print(f"\tWeight init: {self.w}")
		print(f"\tBias   init: {self.b}")
		self.lr = lr
		self.name = name


	def forward(self, inputs):
		self.weighted_sum = np.dot(self.w, inputs) + self.b
		return self.sigmoid(self.weighted_sum)

	# def get_batch(self, d1, d2, GD_batchsize=0.2):

	# 	batchs = range(0, len(d1), int(GD_batchsize * len(d1)))
	# 	batchs_min = batchs[:-1]
	# 	batchs_max = batchs[1:]
	# 	print(f"Batchs:{batchs_min}\n{batchs_max}")

	# 	for bound_min, bound_max in zip(batchs_min, batchs_max):
	# 		yield d1[bound_min:bound_max], d2[bound_min:bound_max]

	def gradient_descent(self, features, target):
		"""
			A l'echelle du model (4 neurons) :

				J(θ)		-> Loss Function
				∂J(θ) / ∂θj	-> Dérivé de la Loss Function en fonction d'une weight j
				hθ(xi)		-> Prediction du model avec matrice de poids θ et features d'entrée xi
				yi			-> Target / Résultat attendu

				J(θ) = −AVERAGE[yi * log(hθ(xi)) + (1 − yi) * log(1 − hθ(xi))]
				∂J(θ) / ∂θj = AVERAGE[ (hθ(xi) − yi)xi ]
		"""

		# print(f"features {features.shape}:\n{features}")
		prediction = self.forward(features)

		loss = -target * math.log(prediction) - (1 - target) * math.log(1 - prediction)
		dloss = (prediction - target)
		dfa = self.sigmoid(self.weighted_sum) * (1 - self.sigmoid(self.weighted_sum))
		dws = features

		# dE/dw = dE/dOut * dOut/dws * dws/dw
		self.w = self.w - self.lr * dws * dfa * dloss
		self.b = self.b - self.lr * 1 * dfa * dloss

		return loss, prediction

	def save_weights(self, file_path):

		with open(file_path, 'a') as f:
			[f.write(f"{w}, ") for w in np.nditer(self.w)]
			f.write(f"{self.b}\n")
