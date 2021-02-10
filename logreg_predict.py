import sys
import numpy as np
from logreg import Logreg

def open_files():
    #Open dataset file
    dataset_file = open(sys.argv[1], 'r')
    lines = dataset_file.read()
    dataset_file.close()

    #Open weights file
    weights_file = open(sys.argv[2], 'r')
    models = weights_file.read()
    weights_file.close()    

    return lines, models

def parse(lines, model):

	# Titles
    del lines[0]
	# \n
    del lines[-1]

	#Save weights
	#Save dataset
	return [[float(x) if x.isnumeric() else 0 for x in neuron.split(',')] for neuron in model.split("\n")],
			[[float(x) if x.isnumeric() else 0 for x in student.split(',')[6:]] for student in lines.split("\n")]


if len(sys.argv) != 3:
	print("2 arguments needed")
	exit(1)


# Parsing
data = open_files()

inputs, weights = parse(*data)

print(f"Lines: {inputs}")
print(f"Weights: {weights}")



# Create and fill answers file "houses"
houses_file = open("houses.csv", 'w')
houses_file.write("Index,Hogwarts House\n")
for i, student in enumerate(students):
    houses_file.write(str(i))
    houses_file.write(",")
    houses_file.write(sorting_hat(student))
    print("\n")
houses_file.close()
