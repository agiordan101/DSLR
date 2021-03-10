import sys
import plotly.express as px
from plotly.offline import plot

# Protection
if len(sys.argv) != 2:
	print("1 argument needed: dataset")
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

#Get marks of all students
astronomy = []
defense = []
for line in dataset:
    #print("Convert : ", line[7])
    #print("Convert : ", line[9])
    if (line[7] != "" and line[9] != ""):
        astronomy.append(float(line[7]))
        defense.append(float(line[9]))

fig = px.scatter(x=astronomy, y=defense)
fig.update_layout(
    title="Two similar lessons",
    xaxis_title="Astronomy",
    yaxis_title="Defense Against Dark Art",
)
plot(fig)
