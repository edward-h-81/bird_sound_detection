import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.utils import shuffle

DATA_PATH_TRAIN_VAL = "hold-out_warblr_train_val_114_32mels_filenames.json"
DATA_PATH_TEST = "hold-out_warblr_test_114_32mels_filenames.json"

def load_data(data_path_train_val, data_path_test):

    with open(data_path_train_val, "r") as fp:
        data = json.load(fp)

    X = np.array(data["log mel spectrogram"])
    y = np.array(data["labels"])

    with open(data_path_test, "r") as fp:
        data = json.load(fp)

    X_test = np.array(data["log mel spectrogram"])
    y_test = np.array(data["labels"])
    z_test = np.array(data["filenames"])

    return X, y, X_test, y_test, z_test

def prepare_datasets(validation_size):
    # load data
    X, y, X_test, y_test, z_test = load_data(DATA_PATH_TRAIN_VAL, DATA_PATH_TEST)

    # shuffle test data
    X_test, y_test, z_test = shuffle(X_test, y_test, z_test, random_state=2)

    # create the train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, random_state=2)


    # TF expects a 3D array -> (# time bins, # mel bands, # colours)
    X_train = X_train[..., np.newaxis] # 4D array -> (# samples, # time bins, # mel bands, # colours)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]


    return X_train, X_validation, X_test, y_train, y_validation, y_test, z_test

def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed to dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(2, activation='softmax'))

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
    X_train, X_validation, X_test, y_train, y_validation, y_test, z_test = prepare_datasets(0.2)

    # build the CNN
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the CNN
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=10)

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