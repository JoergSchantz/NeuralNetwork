# a one layer perceptron for multinomial regression
import numpy as np

class Olp:
    def __init__(self):
        self.n_samples = 0
        self.n_features = 0
        self.n_classes = 0
        self.weights = np.array([])
        self.bias = np.array([])

    # fitting via gradient descent
    def fit(self, inputs, target, stop = 0.05):
        self.n_samples, self.n_features = inputs.shape
        self.n_classes = len(np.unique(target))
        self.weights = np.random.rand(self.n_features, self.n_classes)
        
        # adjust inputs and weights for intercept
        self.weights = np.vstack((np.zeros(self.n_classes), self.weights))
        inter = np.ones(self.n_samples).reshape(self.n_samples, 1)
        inputs_and_inter = np.hstack((inter, inputs))
        
        # initial step
        self.softmax(inputs_and_inter)
        current_loss = self.loss(target)
        print("Current loss: ", current_loss)
        
        # GD
        while current_loss > stop:
            dW = self.gradient(inputs_and_inter, target)
            self.weights -= 0.001 * dW
            self.softmax(inputs_and_inter)
            current_loss = self.loss(target)
            print("Current loss: ", current_loss)
            
    # softmax activation function for the layer
    def softmax(self, inputs):
        exp_scores = np.exp(inputs.dot(self.weights)) # calculate scores
        self.output = exp_scores / np.sum(exp_scores, 1, keepdims = True)

    # cross entropy loss
    def loss(self, target):
        logs = np.log(self.output)
        logLoss = -logs[range(self.n_samples), target]

        return np.sum(logLoss) / self.n_samples
    
    # derivative of loss with respect to weights
    def gradient(self, inputs, target):
        # subtract 1 from correct classes
        self.output[np.arange(self.n_samples), target] -= 1
        
        return inputs.T.dot(self.output)

    def predict(self):
        return np.argmax(self.output, 1)