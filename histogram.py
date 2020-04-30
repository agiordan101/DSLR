import sys
import plotly.graph_objects as go
from plotly.offline import plot

#Open dataset
dataset_file = open(sys.argv[1], "r")
lines = dataset_file.read().split('\n')

#Save dataset by students
features = lines[0].split(",")
del lines[0]
del lines[-1]
dataset = []
for i in lines:
    dataset.append(i.split(","))

#Save grade of lesson "Care of Magical Creatures" by houses
gryffindor = []
hufflepuff = []
ravenclaw = []
slytherin = []
for student in dataset:
    if (student[1] == "Gryffindor"):
        gryffindor.append(student[16])
    elif (student[1] == "Hufflepuff"):
        hufflepuff.append(student[16])
    elif (student[1] == "Ravenclaw"):
        ravenclaw.append(student[16])
    else:
        slytherin.append(student[16])

#Add them
fig = go.Figure()
fig.add_trace(go.Histogram(x=gryffindor, name="Gryffindor"))
fig.add_trace(go.Histogram(x=hufflepuff, name="Hufflepuff"))
fig.add_trace(go.Histogram(x=ravenclaw, name="Ravenclaw"))
fig.add_trace(go.Histogram(x=slytherin, name="Slytherin"))

fig.update_layout(
    title="Lesson <Care of Magical Creatures>",
    xaxis_title="Students grades",
    yaxis_title="Number of students",
    barmode='overlay', #Histogram mode rectangle
)
fig.update_traces(opacity=0.5)
plot(fig)
