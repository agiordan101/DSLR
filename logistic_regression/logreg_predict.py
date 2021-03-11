import sys
import numpy as np
#from logreg import Logreg
from logreg import *
from DataProcessing import *

# def parse(dataset_path, weights_path):
	
# 	#Open dataset file
# 	dataset_file = open(dataset_path, 'r')
# 	features = dataset_file.read()
# 	dataset_file.close()

# 	#Open weights file
# 	weights_file = open(weights_path, 'r')
# 	model = weights_file.read()
# 	weights_file.close()

# 	#Save weights as 
# 	#Save dataset
# 	return np.array([[float(x) if x else 0 for x in student.split(',')[6:]] for student in features.split("\n")[1:-1]]), [[float(x) if x else 0 for x in neuron.split(',')] for neuron in model[:-1].split("\n")]

def parse(dataset_path, weights_path):

	#Open dataset file
	dataset_file = open(dataset_path, 'r')
	features_str = dataset_file.read()
	dataset_file.close()

	# Init data structure
	features = {}
	for feature in columns_name:
		features[feature] = []

	# Fill
	for student_str in features_str.split("\n")[1:-1]:
		student_strlst = student_str.split(',')

		for feature, i in pertinent_features.items():
			features[feature].append(float(student_strlst[i]) if student_strlst[i] else 0)

	#Open weights file
	weights_file = open(weights_path, 'r')
	model = weights_file.read()
	weights_file.close()

	model = [[float(x) if x else 0 for x in neuron.split(',')] for neuron in model[:-1].split("\n")]

	# print(f"features: {features}\n")
	return features, model


# Protection
if len(sys.argv) != 3:
	print("2 arguments needed: dataset weights")
	exit(1)

# Parsing
test_dataset, model = parse(sys.argv[1], sys.argv[2])

dataProcessing = DataProcessing(test_dataset, columns=columns_name)
dataProcessing.normalize()
test_dataset = dataProcessing.get_data(data_type="2d_array")

print(f"features ({len(test_dataset)} features) : {test_dataset}")
print(f"Weights: {model}")

# Create model with weights already trainned
model = [Logreg(len(weights), weights=weights) for weights in model]

# Create answers file "houses"
houses_file = open("houses.csv", 'w')
houses_file.write("Index,Hogwarts House\n")

for i, features in enumerate(test_dataset):

	houses_file.write(str(i))
	houses_file.write(",")

	predictions = [neuron.forward(features) for neuron in model]
	house = house_matrix[np.argmax(predictions)]

	# if i == 0:
	# 	print(f"{i} ->\t{features}")
	print(f"Prediction:\t{predictions}")
	print(f"House:\t{house}")

	houses_file.write(house)
	houses_file.write('\n')

houses_file.close()
