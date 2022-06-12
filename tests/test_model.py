import pandas as pd
import numpy as np
from joblib import load

from src.basic_cleaning import clean_dataset
from src.ml.data import process_data
from src.ml.model import inference


def test_clear_data():

    df = pd.read_csv("tests/data/census.csv", skipinitialspace=True)
    df = clean_dataset(df)

    assert df.shape == df.dropna().shape
    assert '?' not in df.values
    assert "fnlgt" not in df.columns
    assert "education-num" not in df.columns
    assert "capital-gain" not in df.columns
    assert "capital-loss" not in df.columns


def test_process_data():
    """
    Check split have same number of rows for X and y
    """
    data = pd.read_csv("data/cleaned/census.csv")

    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

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

    X_test, y_test, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_inference_above():
    """
    Check inference performance
    """
    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    array = np.array([[
                     32,
                     "Private",
                     "Some-college",
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Husband",
                     "Black",
                     "Male",
                     80,
                     "United-States"
                     ]])
    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

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

    X, _, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"


def test_inference_below():
    """
    Check inference performance
    """
    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    array = np.array([[
                     19,
                     "Private",
                     "HS-grad",
                     "Never-married",
                     "Own-child",
                     "Husband",
                     "Black",
                     "Male",
                     40,
                     "United-States"
                     ]])

    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

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

    X, _, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"
