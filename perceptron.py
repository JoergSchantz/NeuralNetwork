# a one neuron perceptron for logistic regression

# import libs
import numpy as np

# define perceptron class
class perceptron:
    def __init__(self):
        pass
    
    # estimate weights via OLS
    def fit(self, features, target):
        
        r, c = features.shape
        # features' first row is the bias
        self.inputs = np.hstack((np.ones([r,1]), features))
        
        # perform OLS
        T = np.transpose(self.inputs)
        A = T @ self.inputs
        self.weights = (np.linalg.inv(A) @ T) @ target
    
    def forward(self):
        self.scores =  np.dot(self.inputs, self.weights)
        
    # response/ activation function is logit function
    def response(self):
        z = np.exp(self.scores)
        self.responses = z / (1 + z)
