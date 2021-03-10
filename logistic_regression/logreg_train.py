import sys
import numpy as np
from logreg import Logreg
from DataProcessing import DataProcessing

house_matrix = ["Gryffindor",
				"Hufflepuff",
				"Ravenclaw",
				"Slytherin"]

def house_to_nparray(house):
	target = np.zeros(4)
	target[house_matrix.index(house)] = 1
	return target

def parse(dataset_path):
	
	#Open dataset file
	dataset_file = open(dataset_path, 'r')
	features_str = dataset_file.read()
	dataset_file.close()

	features_lst = features_str.split("\n")
	columns_name = features_lst[0].split(',')[6:]

	inputs = []
	targets = []
	for student_str in features_lst[1:-1]:
		# print(f"Student: {strstudent.split(',')}")
		student_lst = student_str.split(',')

		targets.append(house_to_nparray(student_lst[1]))

		fstudent = []
		for x in student_lst[6:]:
			fstudent.append(float(x) if len(x) else 0)
		# print(f"inputs: {fstudent}\n")
		inputs.append(fstudent)

	return np.array(inputs), np.array(targets), columns_name


# Protection
if len(sys.argv) != 2:
	print("1 argument needed: dataset")
	exit(1)

# Parsing
inputs, targets, columns_name = parse(sys.argv[1])

print(columns_name)
print(len(columns_name))
print(len(columns_name))
print()

dataProcessing = DataProcessing(inputs[:10], columns=columns_name)
dataProcessing.normalize()
inputs = dataProcessing.get_data()

print(inputs)
print(targets)

# Create model with random weights
models = [Logreg(len(columns_name), name=name) for name in house_matrix]

for epoch in range(2):
	print(f"\n--- EPOCH {epoch} ---\n")
	for i, model in enumerate(models):
		print(f"Train {model.name}...")
		for features, target in zip(inputs[:1], targets[:1]):
			model.gradient_descent(features, target[i])

		