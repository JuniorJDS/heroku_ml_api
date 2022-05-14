import os
from fastapi import FastAPI
from schema import ModelInput
from joblib import load
import numpy as np
import pandas as pd
from src.ml.data import process_data
from src.ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()


@app.get("/")
async def status_check():
    return {"message": "Welcome to the ML Heroku API"}


@app.post("/")
async def post_inference(user_data: ModelInput):


    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    array = np.array([[
                     user_data.age,
                     user_data.workclass,
                     user_data.education,
                     user_data.maritalStatus,
                     user_data.occupation,
                     user_data.relationship,
                     user_data.race,
                     user_data.sex,
                     user_data.hoursPerWeek,
                     user_data.nativeCountry
                     ]])
    df = pd.DataFrame(data=array, columns=[
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
                df,
                categorical_features=cat_features,
                encoder=encoder, 
                lb=lb, 
                training=False
    )

    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]

    return {"prediction": y}
