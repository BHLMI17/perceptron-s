import pandas as pd
from sklearn.utils import shuffle
import numpy as np

#load the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#Shuffle the data
df = shuffle(df)

df.head()