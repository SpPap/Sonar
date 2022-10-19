from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


def create_model(x_train, y_train):

    # initialising the ANN
    model = Sequential()

    # adding the input layer and first hidden layer
    model.add(Dense(30, activation='relu', input_dim=60))
    model.add(Dropout(rate=0.1))

    # second  hidden layer
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(rate=0.1))

    # third hidden layer
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(rate=0.2))

    # output layer
    model.add(Dense(1, activation='sigmoid'))

    # compiling the ANN
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    model.fit(x_train, y_train, batch_size=12, epochs=100)

    return model
