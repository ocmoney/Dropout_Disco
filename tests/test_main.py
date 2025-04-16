import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.text == '"ok"'

def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()

def test_how_many_upvotes():
    test_post = {"title": "Test post title"}
    response = client.post("/how_many_upvotes", json=test_post)
    assert response.status_code == 200
    assert "upvotes" in response.json()

def test_logs():
    response = client.get("/logs")
    assert response.status_code == 200
    assert "logs" in response.json() 