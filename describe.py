import sys
import math

def is_number(elem):
    if (elem == ''):
        return False
    n_dot = 0
    first = 1
    for i in str(elem):
        if (i == '.'):
            if (n_dot == 0):
                n_dot = 1
                continue
            else:
                return False
        if (i == '-'):
            if (first == 0):
                return False
            else:
                continue
        if (i.isdigit() == False):
            return False
        first = 0
    return True

def average(feature):
    sumValue = 0
    for value in feature:
        sumValue += value
    return sumValue / len(feature)


def standarDeviation(feature): # = numpy.std
    tmp = feature[:]
    mean = average(tmp)
    for i in range(len(tmp)):
        tmp[i] = abs(tmp[i] - mean)
        tmp[i] *= tmp[i]
    return math.sqrt(average(tmp))
"""
def standarDeviation(feature): #Ecart type
    tmp = feature[:]
    mean = average(tmp)
    for i in range(len(tmp)):
        tmp[i] = abs(tmp[i] - mean)
    return average(tmp)
"""
def minValue(feature):
    tmp = feature[0]
    for value in feature:
        if (value < tmp):
            tmp = value
    return tmp

def quartile(feature, n):
    length = len(feature)
    for i in range(length):
        if (i > n * length / 4):
            if (i == 1):
                return feature[0]
            else:
                return feature[i - 2]
    return feature

def maxValue(feature):
    tmp = feature[0]
    for value in feature:
        if (value > tmp):
            tmp = value
    return tmp

#Main code

#Open dataset
dataset_file = open(sys.argv[1], "r")
lines = dataset_file.read().split('\n')

#Save dataset
features = lines[0].split(",")
del lines[0]
del lines[-1]
dataset = []
for i in lines:
    dataset.append(i.split(","))

#Save data by features and sort them
data = []
for i in range(len(features)):
    tmp = []
    for student in dataset:
        if (student[i] != ''):
            if (is_number(student[i])):
                tmp.append(float(student[i]))
            else:
                tmp.append(student[i])
    tmp.sort()
    data.append(tmp)

"""
print(data[0])
print(len(data[0]))
print(average(data[0]))
print(standarDeviation(data[0]))
print(minValue(data[0]))
print(quartile(data[0], 1))
print(quartile(data[0], 2))
print(quartile(data[0], 3))
print(maxValue(data[0]))
"""

#Use data
count = []
mean = []
std = []
mini = []
Q1 = []
Q2 = []
Q3 = []
maxi = []
for feature in data:
    count.append(len(feature))
    if (is_number(feature[0])):
        mean.append(average(feature))
        std.append(standarDeviation(feature))
    mini.append(minValue(feature))
    Q1.append(quartile(feature, 1))
    Q2.append(quartile(feature, 2))
    Q3.append(quartile(feature, 3))
    maxi.append(maxValue(feature))        

"""
#Print data
print(count)
print(mean)
print(std)
print(mini)
print(Q1)
print(Q2)
print(Q3)
print(maxi)
print("\n\n")
"""
"""
print("count\t\tmean\t\tstd\t\tmin\t\t25%\t\t50%\t\t75%\t\tmax")
for i in range(len(features)):
    print(features[i], end="\t\t")
    print(count[i], end="\t\t")
    print(mean[i], end="\t\t")
    print(std[i], end="\t\t")
    print(mini[i], end="\t\t")
    print(Q1[i], end="\t\t")
    print(Q2[i], end="\t\t")
    print(Q3[i], end="\t\t")
    print(maxi[i])
"""
for feature in features:
    print("%22s"%feature, end="")
for value in count:
    print("%22.2f"%value, end="")
for value in mean:
    print("%22.2f"%value, end="")
for value in std:
    print("%22.2f"%value, end="")
for value in mini:
    print("%22.2f"%value, end="")
for value in Q1:
    print("%22.2f"%value, end="")
for value in Q2:
    print("%22.2f"%value, end="")
for value in Q3:
    print("%22.2f"%value, end="")
for value in maxi:
    print("%22.2f"%value, end="")






















