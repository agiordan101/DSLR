import sys
import numpy as np
from logreg import Logreg

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
	features = dataset_file.read()
	dataset_file.close()

	inputs = []
	targets = []
	for student_str in features.split("\n")[1:-1]:
		# print(f"Student: {strstudent.split(',')}")
		student_lst = student_str.split(',')

		targets.append(house_to_nparray(student_lst[1]))

		fstudent = []
		for x in student_lst[6:]:
			fstudent.append(float(x) if len(x) else 0)
		# print(f"inputs: {fstudent}\n")
		inputs.append(fstudent)

	return np.array(inputs), np.array(targets)


# Protection
if len(sys.argv) != 2:
	print("2 arguments needed: dataset weights")
	exit(1)

# Parsing
inputs, targets = parse(sys.argv[1])

print(inputs)
print(targets)


