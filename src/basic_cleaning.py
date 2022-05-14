"""
Cleaning Data

Author: Junior J
"""
import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder


def clean_dataset(df: pd.DataFrame):

    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)

    df.replace({'?':np.nan},inplace=True)
    df.dropna(inplace=True)

    # df['salary'] = LabelEncoder().fit_transform((df['salary']))
    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)

    return df


def clear():

    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = clean_dataset(df)
    df.to_csv("data/cleaned/census_cleaned.csv", index=False)
