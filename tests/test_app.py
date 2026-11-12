"""Smoke tests: the app starts, pages render, inputs are validated and the model predicts."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import app as qm  # noqa: E402

GOOD = {"pH": 6.6, "Temperature": 35, "Taste": 1, "Odor": 0, "Fat": 1, "Turbidity": 0, "Colour": 254}


@pytest.fixture()
def client():
    qm.app.config.update(TESTING=True)
    with qm.app.test_client() as c:
        yield c


@pytest.mark.parametrize("path", ["/", "/models", "/about", "/auth"])
def test_pages_render(client, path):
    assert client.get(path).status_code == 200


def test_valid_inputs_pass_validation():
    values, errors = qm.parse_inputs({k: str(v) for k, v in GOOD.items()})
    assert errors == {}
    assert set(values) == set(qm.FIELDS)


def test_invalid_inputs_are_reported():
    bad = dict(GOOD, pH="abc", Taste=3)
    _, errors = qm.parse_inputs(bad)
    assert "pH" in errors and "Taste" in errors


def test_model_predicts_a_known_grade():
    values, _ = qm.parse_inputs(GOOD)
    result = qm.predict(values)
    assert result["grade"] in qm.GRADES
    assert 0 <= result["confidence"] <= 100  # percent
    assert abs(sum(result["probabilities"].values()) - 100) < 1


def test_analyser_and_api_require_login(client):
    assert client.get("/analyse").status_code in (302, 401, 403)
    r = client.post("/api/predict", json=GOOD)
    assert r.status_code in (302, 401, 403)
