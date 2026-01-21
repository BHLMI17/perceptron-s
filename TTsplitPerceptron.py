import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import Perceptron as p

#load the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#Shuffle the data
df = shuffle(df)

# df.head()


X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values




# print(X[0:5])
# print(y[0:5])




# 75% for train and 25% for test
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

# Encoding the labels: 1 for setosa, -1 otherwise
train_labels = np.where(train_labels == 'Iris-setosa', 1, -1)
test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

print('Train data:', train_data[0:2])
print('Train labels:', train_labels[0:5])

print('Test data:', test_data[0:2])
print('Test labels:', test_labels[0:5])


perceptron = p.Perceptron(eta=0.1, n_iter=10)
perceptron.fit(train_data, train_labels)

test_preds = perceptron.predict(test_data)
print(test_preds)


accuracy = accuracy_score(test_preds, test_labels)
print('Accuracy:', round(accuracy, 2) * 100, "%")