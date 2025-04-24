# import necessary libs
from numpy import exp, array, random, dot, append, zeros
from neuron import NeuralNetwork
# import data set 'iris'
from sklearn.datasets import load_iris


test = zeros([1,4])
for i in range(3):
    test = append(test, dot([[0, 0, 1, 1,0], [1, 1, 1, 1,0], [1, 0, 1, 1,0], [0, 1, 1, 0,1]], 2 * random.random((5,1)) - 1).T, axis = 0)
test = test[1:,:]

s = exp(test.astype(float))
ss = sum(s)
D = s / 4*ss
D[[0,0,1,0],range(4)] -= 1.0/4

print(D)
2 * random.random((3,1)) - 1
