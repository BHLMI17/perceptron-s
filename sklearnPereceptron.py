from sklearn.linear_model import Perceptron
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)

#setting the x (in) and y(out) values, which are all the rows from column 0-3 and 4 respectively
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# Encode BEFORE splitting
y = np.where(y == 'Iris-setosa', 1, -1)

#setting the values, training data, testing data, training labels and testing labels
#we make it so that the test size is 1/4 of the entire size of the data set that we are given
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

#param 1: random state: value used to introduce rng into the perceptron in 3 places:
#  weight initialization, 
# data shuffling (different data order), 
# and train test split, to make sure if you start at any point, it will still work
#param2: max iterations - how many times the perceptron sweeps through the training data set
#param3: tol: stopping tolerance, how close the perceptron gets to the correct answer for it to stop early
perceptron = Perceptron(random_state=42, max_iter=10, tol=0.001)
perceptron.fit(train_data, train_labels)

test_preds = perceptron.predict(test_data)

accuracy = accuracy_score(test_labels, test_preds)
print("Accuracy:", round(accuracy * 100, 2), "%")