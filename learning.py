import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def learn(x, y):
    # Encoding the Independent Variable y // R->1 M->0
    lb_y = LabelBinarizer()
    y[:] = lb_y.fit_transform(y[:])

    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

    """
    #Scaling (already scaled dataset)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    """

    x_test = np.asarray(x_test).astype('float64')
    x_train = np.asarray(x_train).astype('float64')

    y_test = np.asarray(y_test).astype('float64')
    y_train = np.asarray(y_train).astype('float64')

    return x_test, x_train, y_test, y_train
