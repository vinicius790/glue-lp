"""API route smoke — skip if fastapi not installed."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from glue_lp.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_protocol_invariant(client):
    r = client.get("/protocol/invariant", params={"seed": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"]["invariant_held"] is True
    assert body["leaky"]["leakage"] is True


def test_protocol_check_alias(client):
    r = client.get("/protocol/check", params={"seed": 7})
    assert r.status_code == 200
    assert r.json()["valid"]["invariant_held"] is True


def test_experiments_summary(client):
    r = client.get("/experiments/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] >= 3
    assert isinstance(body["experiments"], list)


def test_experiments_list(client):
    r = client.get("/experiments")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) >= 3


def test_experiment_serie1(client):
    r = client.get("/experiments/serie1")
    assert r.status_code == 200
    assert "E1_gcn_valid_auc" in r.json()["summary"]
