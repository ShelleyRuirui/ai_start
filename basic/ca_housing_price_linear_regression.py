from keras.datasets import california_housing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import keras
from dataclasses import dataclass


@dataclass
class Settings:
    input_features: list[str]
    label_name: str
    learning_rate: float
    batch_size: int
    epochs: int


def print_dataframe(dataframe):
    pd.set_option('display.max_columns', None)
    print('Total number of rows: {0}\n\n'.format(len(dataframe.index)))
    print('\n## Printing example rows')
    print(dataframe[0:5])
    print('\n## Printing statistics')
    print(dataframe.describe(include='all'))
    print('\n## Printing correlation')
    print(dataframe.corr(numeric_only = True))
    fig = px.scatter_matrix(dataframe, dimensions=["PRICE", "MEDIAN_INCOME",
                                                   "MEDIAN_HOUSE_AGE"])
    # fig.show()


def plot_history(history):
    # Get the data from the history object
    loss = history.history['loss']
    rmse = history.history['rmse']
    epochs = range(1, len(loss) + 1)

    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))

    # Plot the training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, rmse, 'bo-', label='RMSE')
    plt.title('Training rmse')
    plt.xlabel('Epochs')
    plt.ylabel('rmse')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjusts plots to prevent overlap
    plt.show()


def convert_datasets_to_dataframe(x_dataset, y_dataset):
    x_dataset_dataframe = pd.DataFrame(x_dataset)
    
    x_dataset_dataframe.columns = [ 'LONG', 'LAT', 'MEDIAN_HOUSE_AGE',
                                 'TOTAL_ROOM', 'TOTAL_BEDRM', 'POPULATION',
                                 'HOUSE_HOLD', 'MEDIAN_INCOME']
    dataframe = x_dataset_dataframe.assign(PRICE=y_dataset)
    # print_dataframe(dataframe)
    return dataframe


def convert_dataframe_to_features_and_labels(
        dataset:pd.DataFrame, input_features:list[str], label_name: str):
    features = {name: dataset[name].values for name in input_features}
    label = dataset[label_name].values
    return features, label

def load_and_describe_data():
    # Load the full dataset with a 25% test split and a fixed random seed
    (x_train, y_train), (x_test, y_test) = california_housing.load_data(
        version="large",
        path="california_housing.npz",
        test_split=0.25,
        seed=42
    )
    print(f"Type of training dataset x is : {type(x_train)}")
    print(f"Shape of training features: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of test features: {x_test.shape}")
    print(f"Shape of test labels: {y_test.shape}")

    train_dataframe = convert_datasets_to_dataframe(x_train, y_train)
    test_dataframe = convert_datasets_to_dataframe(x_test, y_test)
    return train_dataframe, test_dataframe


def create_model(settings: Settings,
                 metrics: list[keras.metrics.Metric]) -> keras.Model:
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    all_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(all_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(
        learning_rate=settings.learning_rate), loss="mean_squared_error",
        metrics=metrics)
    return model


def train_model(name: str, model: keras.Model, dataset: pd.DataFrame,
                settings: Settings):
    (features, label) = convert_dataframe_to_features_and_labels(
        dataset, settings.input_features, settings.label_name)
    history = model.fit(x=features, y=label, batch_size=settings.batch_size,
                        epochs=settings.epochs)
    print(f"Training {name} complete")
    print(history.history)
    plot_history(history)
    return history


def train_first_model(dataset):
    input_features=['TOTAL_ROOM']
    label_name=['PRICE']
    settings = Settings(learning_rate=0.001, epochs=20, batch_size=50,
                        input_features=input_features, label_name=label_name)
    metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]
    model = create_model(settings, metrics)
    train_model('one_feature', model, dataset, settings)
    return model, settings


def evaluate(model, dataset, settings):
    (features, labels) = convert_dataframe_to_features_and_labels(
        dataset, settings.input_features, settings.label_name)
    # Evaluate the model on the test data
    # This returns the loss and any other metrics (e.g., accuracy or rmse)
    test_results = model.evaluate(x=features, y=labels, verbose=0)

    # The output is a list. The first item is the loss.
    # The following items are the metrics in the order you defined them in model.compile().
    # You can get the metric names from the model's attributes.
    print(f"Test Loss: {test_results[0]:.4f}")

    # If you had other metrics like 'accuracy', you can access them like this:
    # print(f"Test Accuracy: {test_results[1]:.4f}")



def train():
    (train_dataset, test_dataset) = load_and_describe_data()
    (model, settings) = train_first_model(train_dataset)
    evaluate(model, test_dataset, settings)


if __name__ == '__main__':
    train()