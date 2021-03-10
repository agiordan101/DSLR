import sys
import numpy as np
from logreg import Logreg
from DataProcessing import DataProcessing

house_matrix = ["Gryffindor",
				"Hufflepuff",
				"Ravenclaw",
				"Slytherin"]

pertinent_features = {'Astronomy': 7,
						'Herbology': 8,
						'Defense Against the Dark Arts': 9,
						'Ancient Runes': 12}

def house_to_nparray(house):
	target = np.zeros(4)
	target[house_matrix.index(house)] = 1
	return target

def parse(dataset_path):


	#Open dataset file
	dataset_file = open(dataset_path, 'r')
	features_str = dataset_file.read()
	dataset_file.close()

	# Init data structure
	targets = []
	features = {}
	for feature in pertinent_features.keys():
		features[feature] = []

	# Fill
	for student_str in features_str.split("\n")[1:-1]:
		student_strlst = student_str.split(',')

		targets.append(house_to_nparray(student_strlst[1]))

		for feature, i in pertinent_features.items():
			features[feature].append(float(student_strlst[i]) if len(student_strlst[i]) else 0)

	# print(f"features: {features}\n")
	return features, np.array(targets)


# Protection
if len(sys.argv) != 2:
	print("1 argument needed: dataset")
	exit(1)

# Parsing
features, targets = parse(sys.argv[1])
columns_name = list(pertinent_features.keys())

dataProcessing = DataProcessing(features, columns=columns_name)
dataProcessing.normalize()
inputs = dataProcessing.get_data()

print(f"inputs after data processing:\n{inputs}")
print(f"targets after data processing:\n{targets}")

# Create model with random weights
models = [Logreg(len(columns_name), name=name) for name in house_matrix]


# for epoch in range(5):
epoch = 0
accuracy = 0
while accuracy < 0.98:
	print(f"\n--- EPOCH {epoch} ---\n")

	loss_sum = 0
	accuracy_sum = 0
	for features, target in zip(inputs, targets):

		prediction = []
		for i, model in enumerate(models):
			# print(f"\nTrain {model.name}...")
			l, p = model.gradient_descent(features, target[i])

			loss_sum += l
			prediction.append(p)

		# Right prediction
		if target[np.array(prediction).argmax()]:
			accuracy_sum += 1
	
	epoch += 1
	loss = loss_sum / (len(models) * len(inputs))
	accuracy = accuracy_sum / len(inputs)

	print(f"EPOCH {epoch} -> Loss:     {loss}")
	print(f"EPOCH {epoch} -> Accuracy: {accuracy} ({accuracy_sum}/{len(inputs)})")

print(f"--- TRAIN FINISH --- epoch: {epoch} / accuracy: {accuracy}")

dataProcessing.save_data("ressources/normalization.txt")
[model.save_weights("ressources/weights.txt") for model in models]
