import numpy as np

class Logreg:
    
    def __init__(self, n_inputs, lr, weights=[]):
        self.n_inputs = n_inputs
        self.weights = weights
        self.lr = lr
        
    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))
    
    def train(self, inputs, targets):
        
        dWeights = [0] * self.n_inputs
        for i, x in enumerate(inputs):
            output = self.predict(x)
            for j in self.n_inputs:
                dWeights[j] += (output - targets[i]) * x[j]
        dWeights /= len(inputs)
        self.weights -= dWeights * self.lr
    
    def set_weights(self, weights):
        self.weights = weights
    """
    def get_weights(self):
        return self.weights
    """
    def sigmoid(self, x):
        print(x)
        return 1 / (1 + np.exp(-x))
