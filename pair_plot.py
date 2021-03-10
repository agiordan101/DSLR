import sys
import numpy as np
import plotly.express as px
from plotly.offline import plot

# Protection
if len(sys.argv) != 2:
	print("1 arguments needed: dataset")
	exit(1)

#Open dataset and get lines
dataset_file = open(sys.argv[1], "r")
lines = dataset_file.read().split('\n')
del lines[0]
del lines[-1]

#Parse dataset
dataset = []
for i in lines:
    dataset.append(i.split(","))

#Save data by features
data = []
for i in range(6, len(dataset[0])):
    tmp = []
    for student in dataset:
        if (student[i] == ''):
            tmp.append(np.nan)
        else:
            tmp.append(float(student[i]))
    data.append(tmp)

df = dict(Arithmancy=data[0],
          Astronomy=data[1],
          Herbology=data[2],
          Defense_Against_the_Dark_Arts=data[3],
          Divination=data[4],
          Muggle_Studies=data[5],
          Ancient_Runes=data[6],
          History_of_Magic=data[7],
          Transfiguration=data[8],
          Potions=data[9],
          Care_of_Magical_Creatures=data[10],
          Charms=data[11],
          Flying=data[12])
fig = px.scatter_matrix(df,
                       dimensions=["Arithmancy",
                                   "Astronomy",
                                   "Herbology",
                                   "Defense_Against_the_Dark_Arts",
                                   "Divination",
                                   "Muggle_Studies",
                                   "Ancient_Runes",
                                   "History_of_Magic",
                                   "Transfiguration",
                                   "Potions",
                                   "Care_of_Magical_Creatures",
                                   "Charms",
                                   "Flying"], 
                       title="Scatter matrix of students grades")
plot(fig)
