import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('sonar.csv')

#Shuffle data (rows)
data = data.sample(frac=0.5, replace=True, random_state=1)


dataset = data.values

#dataset.head()
X = dataset[:, :-1]
y = dataset[:, -1]

#reshape u to be a 2D array
y = y.reshape(len(y), 1)

# Encoding the Independent Variable y // R->1 M->0
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
labelbinarizer_y = LabelBinarizer()
y[:] = labelbinarizer_y.fit_transform(y[:])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)

"""
#Scaling (already scaled dataset)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

X = np.asarray(X).astype('float64')
y = np.asarray(y).astype('float64')

X_test = np.asarray(X_test).astype('float64')
X_train = np.asarray(X_train).astype('float64')

y_test = np.asarray(y_test).astype('float64')
y_train = np.asarray(y_train).astype('float64')

#importing Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


#initialising the ANN
model = Sequential()

#adding the input layer and first hidden layer
model.add(Dense(30, activation = 'relu', input_dim = 60))
model.add(Dropout(rate = 0.1))

#second  hidden layer
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(rate = 0.1))

#third hidden layer
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(rate = 0.2))

#output layer
model.add(Dense(1, activation = 'sigmoid')) 

#compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 12, epochs = 100)

# Predicting the Test set results
score = model.evaluate(X_test, y_test, batch_size = 12)

y_pred = model.predict(X_test)
y_pred = 1*(y_pred > 0.5)
max(y_pred)


#Confusion Matrix for validation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

prediction = np.array([[0.0108, 0.0086, 0.0058 ,0.0460 ,0.0752 ,0.0887
, 0.1015 ,0.0494 ,0.0472 ,0.0393 ,0.1106 ,0.1412
, 0.2202 ,0.2976 ,0.4116 ,0.4754 ,0.5390 ,0.6279
, 0.7060 ,0.7918 ,0.9493 ,1.0000 ,0.9645 ,0.9432
, 0.8658 ,0.7895 ,0.6501 ,0.4492 ,0.4739 ,0.6153
, 0.4929 ,0.3195 ,0.3735 ,0.3336 ,0.1052 ,0.0671
, 0.0379 ,0.0461 ,0.1694 ,0.2169 ,0.1677 ,0.0644
, 0.0159 ,0.0778 ,0.0653 ,0.0210 ,0.0509 ,0.0387
, 0.0262 ,0.0101 ,0.0161 ,0.0029 ,0.0078 ,0.0114
, 0.0083 ,0.0058 ,0.0003 ,0.0023 ,0.0026 ,0.0027]])
new_prediction = model.predict(prediction)
if new_prediction > 0.5:
    print("It's a Rock")
else:
    print("It's a Mine")



