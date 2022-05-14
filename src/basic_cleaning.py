"""
Cleaning Data

Author: Junior J
"""
import pandas as pd
import numpy as np


def clean_dataset(df: pd.DataFrame):

    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)

    df.replace({'?':np.nan},inplace=True)
    df.dropna(inplace=True)

    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("education-num", axis=1, inplace=True)
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)

    return df


def clear():

    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = clean_dataset(df)
    df.to_csv("data/cleaned/census.csv", index=False)
