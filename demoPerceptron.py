from sklearn import datasets
from perceptron import perceptron


# TEST RUN 
# prepare data for binomial regression
iris = datasets.load_iris()

setosa = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        setosa.append(1)
    else:
        setosa.append(0)

iris['setosa'] = setosa

# init preceptron
logit = perceptron()

# fit weights
logit.fit(iris['data'], iris['setosa'])
# calc scores
logit.forward()
# calc probs
logit.response()
print(logit.responses)


