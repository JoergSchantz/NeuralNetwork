from sklearn import datasets
import olp

# test OLP
iris = datasets.load_iris()

new_layer = olp.Olp()

new_layer.fit(iris['data'], iris['target'])

iris['predictions'] = new_layer.predict()

import matplotlib.pyplot as plt

plt.scatter(iris['data'][:,0],iris['data'][:,1], c = iris['target'])
plt.title('True labels')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
plt.scatter(iris['data'][:,0],iris['data'][:,1], c = iris['predictions'])
plt.title('Predicted labels')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
