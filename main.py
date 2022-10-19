import numpy as np
import pandas as pd
import tensorflow

from sklearn.metrics import confusion_matrix

from learning import learn
from modeling import create_model

# Importing the dataset
data = pd.read_csv('sonar.csv')

# Shuffle data (rows)
data = data.sample(frac=0.5, replace=True, random_state=1)

dataset = data.values

# dataset.head()
x = dataset[:, :-1]
y = dataset[:, -1]

# reshape u to be a 2D array
y = y.reshape(len(y), 1)

# Data learn
x_test, x_train, y_test, y_train = learn(x, y)

# Model creation
model = create_model(x_train, y_train)

# Predicting the Test set results
score = model.evaluate(x_test, y_test, batch_size=12)

y_pred = model.predict(x_test)
y_pred = 1 * (y_pred > 0.5)
max(y_pred)

# Confusion Matrix for validation
cm = confusion_matrix(y_test, y_pred)

prediction = np.array(
    [[
        0.0108, 0.0086, 0.0058, 0.0460, 0.0752, 0.0887,
        0.1015, 0.0494, 0.0472, 0.0393, 0.1106, 0.1412,
        0.2202, 0.2976, 0.4116, 0.4754, 0.5390, 0.6279,
        0.7060, 0.7918, 0.9493, 1.0000, 0.9645, 0.9432,
        0.8658, 0.7895, 0.6501, 0.4492, 0.4739, 0.6153,
        0.4929, 0.3195, 0.3735, 0.3336, 0.1052, 0.0671,
        0.0379, 0.0461, 0.1694, 0.2169, 0.1677, 0.0644,
        0.0159, 0.0778, 0.0653, 0.0210, 0.0509, 0.0387,
        0.0262, 0.0101, 0.0161, 0.0029, 0.0078, 0.0114,
        0.0083, 0.0058, 0.0003, 0.0023, 0.0026, 0.0027,
    ]]
)

new_prediction = model.predict(prediction)

if new_prediction > 0.5:
    print("It's a Rock")
else:
    print("It's a Mine")
