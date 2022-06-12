import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

from src.ml.data import process_data
from src.ml.model import compute_model_metrics


def execute_check_model():

    df = pd.read_csv("data/cleaned/census.csv")
    _, test = train_test_split(df, test_size=0.20)

    trained_model = load("data/model/model.joblib")
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

    with open('data/model/slice_output.txt', 'w') as file:
        for category in cat_features:
            for cls in test[category].unique():
                temp_df = test[test[category] == cls]

                x_test, y_test, _, _ = process_data(
                    temp_df,
                    categorical_features=cat_features, training=False,
                    label="salary", encoder=encoder, lb=lb)

                y_pred = trained_model.predict(x_test)

                prc, rcl, fb = compute_model_metrics(y_test, y_pred)

                metric_info = "[%s]-[%s] Precision: %s " \
                              "Recall: %s FBeta: %s" % (category, cls,
                                                        prc, rcl, fb)
                file.write(metric_info + '\n')
