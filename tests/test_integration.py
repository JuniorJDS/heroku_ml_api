import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    client = TestClient(app)
    return client


def test_status_check__get_status_of_api__expected_200(client):

    request = client.get("/")

    assert request.status_code == 200
    expected = {"message": "Welcome to the ML Heroku API"}
    assert request.json() == expected


def test_post_inference__post_a_valid_data__expected_200(client):

    request = client.post("/", json={
        "age": 19,
        "workclass": "Private",
        "education": "HS-grad",
        "maritalStatus": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })

    assert request.status_code == 200
    assert request.json() == {"prediction": "<=50K"}


def test_post_inference__post_a_second_valid_data__expected_200(client):

    request = client.post("/", json={
        "age": 30,
        "workclass": "Private",
        "education": "Some-college",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 100,
        "nativeCountry": "United-States"
    })

    assert request.status_code == 200
    # modificar
    # assert request.json() == {"prediction": ">50K"}
