import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from src.ml.data import process_data
from src.ml.model import train_model


def execute_train_test_model():
    
    df = pd.read_csv("data/cleaned/census.csv")
    train, _ = train_test_split(df, test_size=0.20)

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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )


    # train and save model
    trained_model = train_model(X_train, y_train)

    dump(trained_model, "data/model/model.joblib")
    dump(encoder, "data/model/encoder.joblib")
    dump(lb, "data/model/lb.joblib")

