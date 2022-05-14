import pandas as pd
import pytest
from src.basic_cleaning import clean_dataset


def test_clear_data():

    df = pd.read_csv("tests/data/census.csv", skipinitialspace=True)
    df = clean_dataset(df)

    assert df.shape == df.dropna().shape
    assert '?' not in df.values
    assert "fnlgt" not in df.columns
    assert "education-num" not in df.columns
    assert "capital-gain" not in df.columns
    assert "capital-loss" not in df.columns
