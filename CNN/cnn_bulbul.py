import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "mixed_114_32mels_filenames.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["log mel spectrogram"])
    y = np.array(data["labels"])
    z = np.array(data["filenames"])
    return X, y, z

def prepare_datasets(test_size, validation_size):

    # load data
    X, y, z = load_data(DATA_PATH)

    # create the train/test split
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=test_size)
    print(len(X_train))
    print(len(X_test))
    print(len(y_train))
    print(len(y_test))
    print(len(z_train))
    print(len(z_test))

    # create the train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    print()
    print(len(X_train))
    print(len(X_validation))
    print(len(y_train))
    print(len(y_validation))

    # TF expects a 3D array -> (# time bins, # mel bands, # colours)
    X_train = X_train[..., np.newaxis] # 4D array -> (# segments, # time bins, # mel bands, # colours)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, z_train, z_test

def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(16, (1, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((1, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 4th conv layer
    model.add(keras.layers.Conv2D(16, (1, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((1, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed to dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(32, activation='relu'))
    # model.add(keras.layers.Dropout(0.3))


    # output layer
    model.add(keras.layers.Dense(2, activation='sigmoid'))

    return model

def predict(model, X, y, z):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    dataLabel = ""
    myPrediction = ""
    if y == 0:
        dataLabel = "BIRD!"
    else:
        dataLabel = "NO BIRD!"

    if predicted_index == 0:
        myPrediction = "BIRD?"
    else:
        myPrediction = "NO BIRD?"

    print("Predicted: {}, Expected: {}, File: {}".format(myPrediction, dataLabel, z))


if __name__ == "__main__":
    # create training, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test, z_train, z_test = prepare_datasets(0.25, 0.2)

    # build the CNN
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the CNN
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    index = 0
    # make prediction on a sample
    while index < len(X_test):
        X = X_test[index]
        y = y_test[index]
        z = z_test[index]
        print("Sample: {}".format(index+1))
        predict(model, X, y, z)
        index += 1