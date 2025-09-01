import numpy as np
import keras
from keras import layers
from PIL import Image


## Prepare data
def load_data():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Print out some examples to see
    print("x_train_type:", type(x_train), "y_train_type", type(y_train))
    print("x_test_type:", type(x_test), "y_test_type", type(y_test))
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    return x_train, y_train, x_test, y_test


def normalize_dataset(dataset):
    dataset = dataset.astype("float") / 255
    dataset = np.expand_dims(dataset, -1)
    return dataset


def train_model():
    (x_train, y_train, x_test, y_test) = load_data()
    x_train = normalize_dataset(x_train)
    x_test = normalize_dataset(x_test)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    print("x_train shape:", x_train.shape)

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    ## Build model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()

    ## Train model
    batch_size = 128
    epochs = 15
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


    ## Evaluate model
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return model


def load_img():
    # Open the image file
    image_path = './aicc/number5.png'
    pil_image = Image.open(image_path)
    new_img = np.array(pil_image)
    new_img = np.expand_dims(new_img, 0)
    new_img = normalize_dataset(new_img)

    # Print the shape to verify
    print(f"Image successfully loaded!")
    print(f"Shape of the NumPy array: {new_img.shape}")
    print(f"Data type of the array: {new_img.dtype}")
    return new_img


def predict(model, new_img):
    predictions = model.predict(new_img)
    found_prediction = False
    print(predictions)
    for index in range(0,9):
        if predictions[0][index] > 0.5:
            found_prediction = True
            print("\n Prediction:", index)
            break
    if not found_prediction:
        print("\n Not found a very likely prediction, predictions:")


if __name__ == '__main__':
    model = train_model()
    new_img = load_img()
    predict(model, new_img)