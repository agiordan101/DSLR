import sys
import matplotlib.pyplot as plt

# Protection
if len(sys.argv) != 2:
	print("1 argument needed: dataset")
	exit(1)

#Open dataset and get lines
dataset_file = open(sys.argv[1], "r")
lines = dataset_file.read().split('\n')

#Save dataset by students
features = lines[0].split(",")
del lines[0]
del lines[-1]
dataset = [i.split(",") for i in lines]

#Save grade of lesson "Care of Magical Creatures" by houses
gryffindor = []
hufflepuff = []
ravenclaw = []
slytherin = []
for student in dataset:
    grade = float(student[16]) if student[16] else 0
    if (student[1] == "Gryffindor"):
        gryffindor.append(grade)
    elif (student[1] == "Hufflepuff"):
        hufflepuff.append(grade)
    elif (student[1] == "Ravenclaw"):
        ravenclaw.append(grade)
    else:
        slytherin.append(grade)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.hist(gryffindor)
ax2.hist(hufflepuff)
ax3.hist(ravenclaw)
ax4.hist(slytherin)
plt.show()
