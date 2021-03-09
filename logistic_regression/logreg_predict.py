import sys
import numpy as np
from logreg import Logreg

house_matrix = ["Gryffindor",
				"Hufflepuff",
				"Ravenclaw",
				"Slytherin"]

def parse(dataset_path, weights_path):
	
	#Open dataset file
	dataset_file = open(dataset_path, 'r')
	features = dataset_file.read()
	dataset_file.close()

	#Open weights file
	weights_file = open(weights_path, 'r')
	model = weights_file.read()
	weights_file.close()

	#Save weights as 
	#Save dataset
	return np.array([[float(x) if len(x) else 0 for x in student.split(',')[6:]] for student in features.split("\n")[1:-1]]), [[float(x) if is_float(x) else 0 for x in neuron.split(',')] for neuron in model[:-1].split("\n")]
# for student in features.split("\n")[1:-1]:
# 	print(f"Student: {student.split(',')[6:]}")
# 	inputs = []
# 	for x in student.split(',')[6:]:
# 		if is_float(x):
# 			n = float(x)
# 		else:
# 			print(f"Not a float: {x}")
# 			n = 0
# 		inputs.append(n)
# 	print(f"inputs: {inputs}\n")


# Protection
if len(sys.argv) != 3:
	print("2 arguments needed: dataset weights")
	exit(1)

# Parsing
inputs, model = parse(sys.argv[1], sys.argv[2])

print(f"features ({len(inputs[0])} features) : {inputs}")
print(f"Weights: {model}")

# Create model with weights already trainned
model = [Logreg(len(weights), weights=weights) for weights in model]

# Create answers file "houses"
houses_file = open("houses.csv", 'w')
houses_file.write("Index,Hogwarts House\n")

for i, features in enumerate(inputs[:1]):

	houses_file.write(str(i))
	houses_file.write(",")

	predictions = [neuron.forward(features[:4]) for neuron in model]
	house = house_matrix[np.argmax(predictions)]

	if i == 0:
		print(f"{i} ->\t{features}")
		print(f"Prediction:\t{predictions}")
		print(f"House:\t{house}")

	houses_file.write(house)
	houses_file.write('\n')

houses_file.close()
