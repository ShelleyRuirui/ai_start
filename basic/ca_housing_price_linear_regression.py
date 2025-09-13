from keras.datasets import california_housing
import numpy as np
import pandas as pd

import plotly.express as px


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
    fig.show()

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

    x_train_dataframe = pd.DataFrame(x_train)
    
    x_train_dataframe.columns = [ 'LONG', 'LAT', 'MEDIAN_HOUSE_AGE',
                                 'TOTAL_ROOM', 'TOTAL_BEDRM', 'POPULATION',
                                 'HOUSE_HOLD', 'MEDIAN_INCOME']
    train_dataframe = x_train_dataframe.assign(PRICE=y_train)
    print_dataframe(train_dataframe)
    return x_train, y_train, x_test, y_test


def train_model():
    (x_train, y_train, x_test, y_test) = load_and_describe_data()


if __name__ == '__main__':
    train_model()