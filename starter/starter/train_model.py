# Script to train machine learning model.
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import sklearn


def ml_pipeline():
    # Add code to load in the data.
    data = pd.read_csv("starter/data/census.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.2)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Train the model
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True)

    model = train_model(X_train, y_train)

    # Saving the encoder and the LabelBinarizer for being used in the API later
    pickle.dump(encoder, open("/starter/model/encoder.pkl", 'wb'))
    pickle.dump(lb, open("/starter/model/label_binarizer.pkl", 'wb'))

    # Validation
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)



if __name__ == "__main__":
    ml_pipeline()
