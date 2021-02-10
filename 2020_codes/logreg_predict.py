import sys
import numpy as np
from logreg_binary import *

def open_files():
    #Open dataset file
    dataset_file = open(sys.argv[1], 'r')
    lines = dataset_file.read().split("\n")
    dataset_file.close()

    #Open weights file
    weights_file = open(sys.argv[2], 'r')
    models = weights_file.read().split("\n\n")
    weights_file.close()    

    return lines, models



def data_processing(lines, models):
    #Save weights
    weights = []
    for model in models:
        weights.append(model.split("\n"))
        for i in range(len(weights[-1])):
            weights[-1][i] = float(weights[-1][i])
    
    #Save dataset
    del lines[0]
    del lines[-1]
    students = []
    for line in lines:
        student = line.split(",")
        student_grades = []
        for i in range(6, len(student)):
            if (student[i] == ""): #Convert missing values -> 0
                student_grades.append(0)
            else:
                student_grades.append(float(student[i]))
        students.append(student_grades)

    #print(weights)
    #print("\n\n\n", students)
    return students, weights



def sorting_hat(inputs):
    house = "Gryffindor\n"
    p = gryffindor.predict(inputs)
    tmp = slytherin.predict(inputs)
    if (tmp > p):
        house = "Slytherin\n"
        p = tmp
    tmp = ravenclaw.predict(inputs)
    if (tmp > p):
        house = "Ravenclaw\n"
        p = tmp
    if (hufflepuff.predict(inputs) > p):
        house = "Hufflepuff\n"
    return (house)



lines, models = open_files()
students, weights = data_processing(lines, models)

#Create models with weights
gryffindor = Logreg(3, 1, weights[0])
slytherin = Logreg(3, 1, weights[1])
ravenclaw = Logreg(3, 1, weights[2])
hufflepuff = Logreg(3, 1, weights[3])

"""
#Remove useless features
"""

#Create and fill answers file "houses"
houses_file = open("houses.csv", 'w')
houses_file.write("Index,Hogwarts House\n")
for i, student in enumerate(students):
    houses_file.write(str(i))
    houses_file.write(",")
    houses_file.write(sorting_hat(student))
    print("\n")
houses_file.close()
