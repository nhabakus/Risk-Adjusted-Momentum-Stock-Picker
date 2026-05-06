"""
model.py

This module builds and trains a neural network model using Keras
to predict whether a stock will have a positive return next week.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


FEATURE_COLUMNS = ["mom_1w", "mom_2w", "mom_4w", "vol_4w", "score"]


def split_train_test_by_date(feature_df, split_date="2024-01-01"):
    """
    Splits the feature data into training and testing sets based on date.

    Args:
        feature_df (pandas.DataFrame): Clean feature dataset.
        split_date (str): Date where test period begins.

    Returns:
        tuple: train_df and test_df.
    """
    train_df = feature_df[feature_df["date"] < split_date].copy()
    test_df = feature_df[feature_df["date"] >= split_date].copy()

    return train_df, test_df


def prepare_train_test_data(train_df, test_df):
    """
    Prepares scaled training and testing data.

    Args:
        train_df (pandas.DataFrame): Training dataset.
        test_df (pandas.DataFrame): Testing dataset.

    Returns:
        tuple: X_train, y_train, X_test, y_test, scaler.
    """
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler


def build_model(input_dim):
    """
    Builds a feedforward neural network for binary classification.

    Args:
        input_dim (int): Number of input features.

    Returns:
        keras.Model: Compiled neural network model.
    """
    model = Sequential()

    model.add(Dense(32, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.10))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train, epochs=10, batch_size=64):
    """
    Trains the neural network.

    Args:
        model (keras.Model): Compiled neural network.
        X_train (numpy.array): Training features.
        y_train (numpy.array): Training targets.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        keras.Model: Trained model.
    """
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model


def predict(model, X):
    """
    Generates model predictions.

    Args:
        model (keras.Model): Trained model.
        X (numpy.array): Feature matrix.

    Returns:
        numpy.array: Predicted probabilities.
    """
    return model.predict(X, verbose=0)