# Python programm to implement a single neuron 

# import necessary libs
from numpy import exp, array, random, dot, log, diag, append

# class to create a single neuron for classification
class Neuron:
    def __init__(self, num_features):
        # ensuring it creates same weights every run
        # random.seed(1)
        # nx1 weight matrix
        self.weight_matrix = 2 * random.random((num_features,1)) - 1
  
    # forward propagation:
    def forward(self, inputs):
        return dot(inputs, self.weight_matrix)
    
# Class to create a neural network for a n-class classification problem
class NeuralNetwork():
    def __init__(self, classes: int, num_features: int):
        self.classes = classes
        self.num_features = num_features
        self.neurons = [Neuron(num_features) for x in range(self.classes)]
    
    # softmax activation function
    def softmax(self, z):
        return exp(z)/sum(exp(z))
    
    # derivate of activation function
    def softmax_grad(self, sm):
        jacobian = diag(sm)
        for i in range(len(jacobian)):
            for k in range(len(jacobian)):
                if i == k:
                    jacobian[i][k] = sm[i] * (1 - sm[i])
                else:
                    jacobian[i][k] = -sm[i] * smk[k]
        return jacobian
    
    # avrg cross entroy loss for softmax
    def cel(self, sm, n_train):
        return sum(-log(sm.T))/n_train
    
    # prediction function
    def predict(self, probabilities):
        
    
    # training NN
    def train(self, train_input, train_output, num_iter):
        num_train = train_input.shape[0]
        for i in range(num_iter):
            # initialize scores vector
            scores = array([[0 for x in range(num_train)]])
            # calc scores for each neuron learning one class
            for k in range(self.classes):
                scores = append(scores, self.neurons[k].forward(train_input).T)
            # drop first irrelevant row of 0s
            scores = scores[1:,:]
            # calc softmax probs
            probs = self.softmax(scores.astype(float))
            for j in 
                error_k = -(train_output[k] * log(output) + (1 - train_output[k])*log(1-output))
                # multiply the error by input and then 
                # by gradient of softmax function to calculate
                # the adjustment needs to be made in weights
                adjust = dot(train_input.T, error * self.sig_deriv(output))
                # adjust weights
                self.weight_matrix += adjust
