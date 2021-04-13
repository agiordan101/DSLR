import sys
import numpy as np
#from logreg import Logreg
from logreg import *
from DataProcessing import *
import matplotlib.pyplot as plt

normalization_path = "ressources/normalization.txt"
weights_path = "ressources/weights.txt"

def house_to_nparray(house):
	target = np.zeros(4)
	target[house_matrix.index(house)] = 1
	return target

def parse(train_dataset):

	#Open dataset file
	dataset_file = open(train_dataset, 'r')
	features_str = dataset_file.read()
	dataset_file.close()

	# Init data structure
	targets = []
	features = {}
	for feature in pertinent_features.keys():
		features[feature] = []

	# Fill
	for student_str in features_str.split("\n")[1:-1]:
		student_strlst = student_str.split(',')

		targets.append(house_to_nparray(student_strlst[1]))

		for feature, i in pertinent_features.items():
			features[feature].append(float(student_strlst[i]) if student_strlst[i] else 0)

	return features, np.array(targets)


# Protection
if len(sys.argv) < 2:
	print("1 argument needed: train_dataset")
	exit(1)

# Option EarlyStopping
delta_min_loss = 0.00001
for i, arg in enumerate(sys.argv):
	if "-earlystopping" in arg.lower() and i + 1 < len(sys.argv):
		delta_min_loss = sys.argv[i + 1]

# Parsing
train_dataset, targets = parse(sys.argv[1])

dataProcessing = DataProcessing(train_dataset, targets, columns=columns_name)
dataProcessing.normalize()
train_dataset, targets = dataProcessing.get_data(
	data_type="2d_array",
	# shuffle=False,
	shuffle=True,
)

if len(train_dataset) != len(targets):
	print(f"len(train_dataset) = {len(train_dataset)}")
	print(f"len(targets) = {len(targets)}")
	exit(0)

# Create model with random weights
models = [Logreg(len(columns_name), name=name) for name in house_matrix]

# for epoch in range(5):
delta_min_loss = 0.00001
epoch = 0
accuracies = []
accuracy = 0
losses = []
loss = 100
last_loss = 1000
while loss < last_loss - delta_min_loss:
	print(f"\n--- EPOCH {epoch} ---\n")

	loss_sum = 0
	accuracy_sum = 0
	fail = 0
	for features, target in zip(train_dataset, targets):

		prediction = []
		for i, model in enumerate(models):
			l, p = model.gradient_descent(features, target[i])

			loss_sum += l
			prediction.append(p)

		# Right prediction
		if target[np.array(prediction).argmax()]:
			accuracy_sum += 1
		else:
			fail += 1

	print(f"len: {len(train_dataset)} / {len(targets)} --- win: {accuracy_sum} --- fail: {fail}")

	last_loss = loss
	loss = loss_sum / (len(models) * len(train_dataset))
	losses.append(loss)
	
	accuracy = accuracy_sum / len(train_dataset)
	accuracies.append(accuracy)

	print(f"EPOCH {epoch} -> Loss:     {loss}")
	print(f"EPOCH {epoch} -> Accuracy: {accuracy} ({accuracy_sum}/{len(train_dataset)})")
	epoch += 1

print(f"--- TRAIN FINISH --- {epoch} epochs / loss: {loss} / accuracy: {accuracy}\n")

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.plot(losses)
ax2.plot(accuracies)
plt.show()

dataProcessing.save_data(normalization_path, normalization=True)

with open(weights_path, 'w+') as f:
	f.close()
	[model.save_weights(weights_path) for model in models]
